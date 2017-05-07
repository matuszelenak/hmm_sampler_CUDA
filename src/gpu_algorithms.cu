#include "gpu_algorithms.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <fstream>

__host__
__device__ double emission_probability(double mean, double stdv, double emission){
	double frac = (emission - mean) / stdv;
	return (1 / (stdv * sqrt(2 * M_PI))) * exp(-0.5 * frac * frac);
}

__host__
__device__ double log_mult(double a, double b){
	if (a == -INFINITY || b == -INFINITY){
		return -INFINITY;
	}
	else{
		return a + b;
	}
}

__host__
__device__ bool log_greater(double a, double b){
	if (a != -INFINITY && b == -INFINITY) return true;
	if (b == -INFINITY) return true;
	return a > b;
}

__global__ void calculate_fw_probs()
{

}

__device__ void traceback_states(int seq_length,
								int num_of_states,
								viterbi_entry *d_viterbi_matrix,
								int *d_state_seq){
	double m = -INFINITY;
	int last_state = 0;
	for (int k = 0; k < num_of_states; k++){
		double v = d_viterbi_matrix[(seq_length-1)*num_of_states + k].prob;
		if (log_greater(v, m)){
			m = v;
			last_state = k;
		}
	}
	d_state_seq[0] = last_state;
	int prev_state = last_state;
	int j = 1;
	for (int i = seq_length - 1; i >= 1; i--){
		int s = d_viterbi_matrix[i*num_of_states + prev_state].state;
		d_state_seq[j++] = s;
		prev_state = s;
	}
}

__host__
void traceback_states_cpu(int seq_length,
								int num_of_states,
								viterbi_entry *viterbi_matrix,
								int *state_seq){
	double m = -INFINITY;
	int last_state = 0;
	for (int k = 0; k < num_of_states; k++){
		printf("At state %d prob %f\n", k, viterbi_matrix[(seq_length-1)*num_of_states + k].prob);
		double v = viterbi_matrix[(seq_length-1)*num_of_states + k].prob;
		if (log_greater(v, m)){
			m = v;
			last_state = k;
		}
		printf("Passed\n");
	}
	printf("Last state is %d", last_state);
	state_seq[0] = last_state;
	int prev_state = last_state;
	int j = 1;
	for (int i = seq_length - 1; i >= 1; i--){
		int s = viterbi_matrix[i*num_of_states + prev_state].state;
		printf("Next state is %d", s);
		state_seq[j++] = s;
		prev_state = s;
	}
}

__global__
void calculate_viterbi_prob(viterbi_entry *d_viterbi_matrix,
							inv_transition *d_inverse_neighbors,
							state_params *d_states,
							int num_of_states,
							int num_of_neighbors,
							int i,
							double emission)
{
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l < num_of_states){
		int maxk = 0;
		double m = -INFINITY;
		double gen_prob;
		for (int j = 0; j < num_of_neighbors; j++){
			inv_transition t = d_inverse_neighbors[l*num_of_neighbors + j];
			gen_prob = log_mult(d_viterbi_matrix[(i - 1)*num_of_states + t.state].prob, t.prob);
			if (log_greater(gen_prob, m)){
				m = gen_prob;
				maxk = t.state;
			}
		}
		
		double em_prob = log(emission_probability(d_states[l].mean, d_states[l].stdv, emission));
		viterbi_entry e;
		e.prob = log_mult(m, em_prob);
		e.state = maxk;
		d_viterbi_matrix[i*num_of_states + l] = e;
	}
}

__global__
void calculate_viterbi_path(int seq_length,
								int num_of_states,
								viterbi_entry *d_viterbi_matrix,
								int *d_state_seq)
{
	traceback_states(seq_length,num_of_states,d_viterbi_matrix,d_state_seq);
}

void save_matrix(viterbi_entry *viterbi_matrix, int seq_length, int num_of_states){
	FILE * pFile;
	pFile = fopen ("viterbi_gpu","w");
	for (int i = 0; i < seq_length; i++){
		for (int j = 0; j < std::max(num_of_states,30); j++){
			fprintf (pFile, "%.2f", viterbi_matrix[i*num_of_states + j].prob);
		}
		fprintf(pFile, "\n");
	}
	fclose (pFile);
}

void gpu_forward_matrix(){
	calculate_fw_probs<<<1, 1>>>();
}

std::vector<int> gpu_viterbi_path(
		std::vector<State> &states,
		std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
		std::vector<double> &event_sequence)
{

	//initialize useful constants
	int num_of_states = inverse_neighbors.size();
	int seq_length = event_sequence.size();
	LogNum init_transition_prob = LogNum(1.0/(double)num_of_states);
	int state_neighbors = inverse_neighbors[0].size();

	//initialize first row of the viterbi matrix (host memory)
	viterbi_entry *viterbi_matrix_row = (viterbi_entry *)malloc(num_of_states * sizeof(viterbi_entry));
	for (int i = 0; i < num_of_states; i++){
		viterbi_entry e;
		e.prob = (init_transition_prob * states[i].get_emission_probability(event_sequence[0])).exponent;
		e.state = 0;
		viterbi_matrix_row[i] = e;
	}

	//convert states to struct type
	state_params *state_p = (state_params *)malloc(num_of_states * sizeof(state_params));
	for (int i = 0; i < num_of_states; i++){
		state_params s;
		s.mean = states[i].corrected_mean;
		s.stdv = states[i].corrected_stdv;
		state_p[i] = s;
	}
	//convert transitions to struct type
	inv_transition *inv_neighbors = (inv_transition *)malloc(num_of_states * state_neighbors *sizeof(inv_transition));
	for (int i = 0; i < num_of_states; i++){
		for (int j = 0; j < inverse_neighbors[i].size(); j++){
			inv_transition t;
			t.state = inverse_neighbors[i][j].first;
			t.prob = (inverse_neighbors[i][j].second).exponent;
			inv_neighbors[i * state_neighbors + j] = t;
		}
	}

	viterbi_entry *d_viterbi_matrix;
	inv_transition *d_inverse_neighbors;
	state_params *d_states;
	int *d_state_seq;

	cudaMalloc((void **)&d_viterbi_matrix, seq_length * num_of_states * sizeof(viterbi_entry));
	cudaMalloc((void **)&d_inverse_neighbors, num_of_states * state_neighbors * sizeof(inv_transition));
	cudaMalloc((void **)&d_states, num_of_states * sizeof(state_params));

	cudaMemcpy(d_viterbi_matrix, viterbi_matrix_row, num_of_states * sizeof(viterbi_entry), cudaMemcpyHostToDevice);
	cudaMemcpy(d_states, state_p, num_of_states * sizeof(state_params), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inverse_neighbors, inv_neighbors, num_of_states * state_neighbors *sizeof(inv_transition), cudaMemcpyHostToDevice);

	int threads_per_block = 512;
	int num_of_blocks = std::max(num_of_states / threads_per_block, 1);

	for (int i = 1; i < seq_length; i++){
		calculate_viterbi_prob<<<num_of_blocks, threads_per_block>>>(
			d_viterbi_matrix,
			d_inverse_neighbors,
			d_states,
			num_of_states,
			state_neighbors,
			i,
			event_sequence[i]);
		cudaDeviceSynchronize();
	}
	
	viterbi_entry *viterbi_matrix = (viterbi_entry *)malloc(num_of_states * seq_length* sizeof(viterbi_entry));

	cudaMemcpy(viterbi_matrix, d_viterbi_matrix, num_of_states * seq_length * sizeof(viterbi_entry), cudaMemcpyDeviceToHost);


	cudaMalloc((void **)&d_state_seq, seq_length * sizeof(int));
	
	calculate_viterbi_path<<<1,1>>>(seq_length,num_of_states,d_viterbi_matrix,d_state_seq);
	cudaDeviceSynchronize();
	int *state_seq = (int *)malloc(seq_length * sizeof(int));

	cudaMemcpy(state_seq, d_state_seq, seq_length * sizeof(int), cudaMemcpyDeviceToHost);
	std::vector<int>res;
	
	copy(state_seq, state_seq + seq_length, std::back_inserter(res));
	std::reverse(res.begin(), res.end());

	save_matrix(viterbi_matrix, seq_length, num_of_states);
	free(viterbi_matrix);
	free(viterbi_matrix_row);
	free(state_p);
	free(inv_neighbors);
	free(state_seq);
	cudaFree(d_viterbi_matrix);
	cudaFree(d_inverse_neighbors);
	cudaFree(d_states);
	cudaFree(d_state_seq);
	return res;
}