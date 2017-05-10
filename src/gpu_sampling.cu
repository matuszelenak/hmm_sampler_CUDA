#include "gpu_algorithms.h"
#include "gpu_utils.h"
#include <cstdio>
#include <algorithm>
#include <cmath>

__global__ void calculate_fw_probs(
		double *d_fw_matrix,
		double *d_prob_matrix,
		inv_transition *d_inverse_neighbors,
		state_params *d_states,
		int num_of_states,
		int max_in_degree,
		int i,
		double emission)
{
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l < num_of_states){
		double em_prob = log(emission_probability(d_states[l].mean, d_states[l].stdv, emission));
		double sum = -INFINITY;
		int actual_neighbors = 0;
		for (int j = 0; j < max_in_degree; j++){
			inv_transition t = d_inverse_neighbors[l*max_in_degree + j];
			int k = t.state;
			if (k == -1) break;
			actual_neighbors ++;
			double gen_prob = log_mult(d_fw_matrix[(i-1) * num_of_states + k], t.prob);
			sum = log_sum(sum, log_mult(gen_prob, em_prob));
			d_prob_matrix[i * num_of_states * max_in_degree
				+ l * max_in_degree + j] = log_mult(gen_prob, em_prob);
		}
		d_fw_matrix[i * num_of_states + l] = sum;
		/*
		if (sum != -INFINITY){
			for (int j = 0; j < actual_neighbors; j++){
				int sh = i * num_of_states * max_in_degree + l * max_in_degree + j;
				d_prob_matrix[sh] = log_div(d_prob_matrix[sh], sum);
			}
			for (int j = actual_neighbors; j < max_in_degree; j++){
				int sh = i * num_of_states * max_in_degree + l * max_in_degree + j;
				d_prob_matrix[sh] = INFINITY;
			}
		}*/
	}
}

__global__ void normalize(double *d_array, int len){
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	double sum = -INFINITY;
	for (int i = 0; i < len; i++){
		if (d_array[l * len + i] == INFINITY) break;
		sum = log_sum(d_array[l * len + i], sum);
	} 
	for (int i = 0; i < len; i++){
		if (d_array[l * len + i] == INFINITY) break;
		d_array[l * len + i] = log_div(d_array[l * len + i], sum);
	}
}

__global__ void prefix_sum(double *d_prob_weights, int max_in_degree){
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	double sum = -INFINITY;
	for (int i = 0; i < max_in_degree; i++){
		if (d_prob_weights[l * max_in_degree + i] == INFINITY) break;
		sum = log_sum(d_prob_weights[l * max_in_degree + i], sum);
		d_prob_weights[l * max_in_degree + i] = sum;
	}
}

void print_prob_matrix(double *prob_matrix, int seq_length, int num_of_states, int max_in_degree){
	for (int i = 0; i < seq_length; i++){
		for (int st = 0; st < num_of_states; st++){
			printf("[");
			for (int ne = 0; ne < max_in_degree; ne++){
				if (prob_matrix[i*num_of_states*max_in_degree + st*max_in_degree + ne] == INFINITY) break;
				printf("%.3f,", exp(prob_matrix[i*num_of_states*max_in_degree + st*max_in_degree + ne]));
			}
			printf("],");
		}
		printf("\n");
	}
}

void gpu_forward_matrix(
		std::vector<State> &states,
		std::vector<double> &event_sequence,
		inv_transition *d_inverse_neighbors,
		int num_of_states,
		int max_in_degree,
		double *d_prob_matrix,
		double *d_last_row_weights)
{

	int seq_length = event_sequence.size();
	LogNum init_transition_prob = LogNum(1.0/(double)num_of_states);

	double *fw_matrix = (double *)malloc(num_of_states * seq_length * sizeof(double));
	double *prob_matrix = (double *)malloc(num_of_states * max_in_degree * seq_length * sizeof(double));
	for (int i = 0; i < num_of_states; i++){
		fw_matrix [i] = (init_transition_prob * states[i].get_emission_probability(event_sequence[0])).exponent;
	}

	//convert states to struct type
	state_params *state_p = (state_params *)malloc(num_of_states * sizeof(state_params));
	for (int i = 0; i < num_of_states; i++){
		state_params s;
		s.mean = states[i].corrected_mean;
		s.stdv = states[i].corrected_stdv;
		state_p[i] = s;
	}

	for (int l = 0; l < num_of_states; l++){
		for (int j = 0; j < max_in_degree; j++){
			prob_matrix[l * max_in_degree + j] = -INFINITY;
		}
	}

	state_params *d_states;
	double *d_fw_matrix;
	cudaMalloc((void **)&d_fw_matrix, seq_length * num_of_states * sizeof(double));
	cudaMalloc((void **)&d_states, num_of_states * sizeof(state_params));

	cudaMemcpy(d_fw_matrix, fw_matrix, num_of_states * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prob_matrix, prob_matrix, num_of_states * max_in_degree * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_states, state_p, num_of_states * sizeof(state_params), cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int num_of_blocks = std::max(num_of_states / threads_per_block, 1);

	cudaEvent_t start_fwm, stop_fwm;
	cudaEventCreate(&start_fwm);
	cudaEventCreate(&stop_fwm);

	cudaEventRecord(start_fwm);
	for (int i = 1; i < seq_length; i++){
		calculate_fw_probs<<<num_of_blocks, threads_per_block>>>(
				d_fw_matrix,
				d_prob_matrix,
				d_inverse_neighbors,
				d_states,
				num_of_states,
				max_in_degree,
				i,
				event_sequence[i]
			);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop_fwm);
	cudaEventSynchronize(stop_fwm);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_fwm, stop_fwm);
	cudaEventDestroy(start_fwm);
	cudaEventDestroy(stop_fwm);

	//normalize the last row:
	double *last_row = (double *)malloc(num_of_states * sizeof(double));
	cudaMemcpy(last_row, d_fw_matrix + (seq_length - 1) * num_of_states,
			  num_of_states * sizeof(double), cudaMemcpyDeviceToHost);
	double sum = -INFINITY;
	for (int i = 0; i < num_of_states; i++){
		sum = log_sum(sum, last_row[i]);
	}
	for (int i = 0; i < num_of_states; i++){
		last_row[i] = log_div(last_row[i], sum);
	}
	cudaMemcpy(d_last_row_weights, last_row, num_of_states * sizeof(double), cudaMemcpyHostToDevice);
	
	/*
	printf("Forward matrix looks like this:\n");
	cudaMemcpy(fw_matrix, d_fw_matrix, num_of_states * seq_length* sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int j = 0; j < num_of_states; j++){
			printf("%.4f ", fw_matrix[i * num_of_states + j]);
		}
		printf("\n");
	}*/
	
	//normalize the prob weights again
	
	threads_per_block = 1024;
	num_of_blocks = std::max(num_of_states * seq_length / threads_per_block, 1);
	normalize<<<num_of_blocks, threads_per_block>>>(d_prob_matrix, max_in_degree);
	cudaDeviceSynchronize();
	
	/*
	cudaMemcpy(prob_matrix, d_prob_matrix, num_of_states * seq_length * max_in_degree * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Prob weights matrix looks like this after being normalized for the second time\n");
	print_prob_matrix(prob_matrix, seq_length, num_of_states, max_in_degree);*/

	cudaFree(d_fw_matrix);
	cudaFree(d_states);

	free(fw_matrix);
	free(prob_matrix);
	free(state_p);
	free(last_row);
}

__device__ int discrete_dist(double *weights, int max_in_degree, curandState *s){
	double val = log(curand_uniform(s));
	for (int i = 0; i < max_in_degree; i++){
		if (weights[i] == INFINITY) return (i - 1);
		if (log_less(val,weights[i])){
			return i;
		}
	}
	return max_in_degree -1;
}

__global__ void backtrack_sample(
								double *d_last_row_weights,
								double *d_prob_weights,
								inv_transition *d_inv_neighbors,
								int seq_length,
								int num_of_states,
								int max_in_degree,
								int num_of_samples,
								int *sample,
								int seed)
{
	unsigned int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_id < num_of_samples){

		curandState s;
		curand_init(seed, sample_id, 0, &s);

		int curr_state = discrete_dist(d_last_row_weights, num_of_states, &s);
		int i = seq_length - 1;
		int rem_length = seq_length - 1;
		while (rem_length >= 0){
			sample[sample_id * seq_length + i] = curr_state;
			i--;
			int shift = rem_length * num_of_states * max_in_degree + curr_state * max_in_degree;
			int next_state_id = discrete_dist(d_prob_weights + shift, max_in_degree, &s);
			curr_state = d_inv_neighbors[curr_state * max_in_degree + next_state_id].state;
			rem_length--;
		}
	}
}

double sumdammit(double a, double b){
	if (a > b){
		return a + log1p(exp(b - a));
	}
	else
	{
		return b + log1p(exp(a - b));
	}
}

std::vector<std::vector<int> > gpu_samples(
	int num_of_samples,
	std::vector<State> &states,
	std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
	int max_in_degree,
	std::vector<double>&event_sequence)
{
	/*
	double test[16] = {-532.268,-368.933,-338.320,-397.375,-322.049,-311.708,-134.850,-340.415,-359.843,-624.111,-733.296,-353.861,-166.022,-140.817,-237.215,-518.579};
	double sum = -INFINITY;
	for (int i = 0; i < 16; i++){
		sum = sumdammit(test[i], sum);
		printf("%.3f ", sum);
	}*/

	cudaEvent_t start_sampling, stop_sampling;
	cudaEventCreate(&start_sampling);
	cudaEventCreate(&stop_sampling);

	int num_of_states = states.size();
	int seq_length = event_sequence.size();

	//convert transitions to struct type
	inv_transition *inv_neighbors = (inv_transition *)malloc(num_of_states * max_in_degree *sizeof(inv_transition));
	for (int i = 0; i < num_of_states; i++){
		for (int j = 0; j < inverse_neighbors[i].size(); j++){
			inv_transition t;
			t.state = inverse_neighbors[i][j].first;
			t.prob = (inverse_neighbors[i][j].second).exponent;
			inv_neighbors[i * max_in_degree + j] = t;
		}
		for (int j = inverse_neighbors[i].size(); j < max_in_degree; j++){
			inv_transition t;
			t.state = -1;
			t.prob = INFINITY;
			inv_neighbors[i * max_in_degree + j] = t;
		}
	}

	double *d_prob_matrix;
	double *d_last_row_weights;
	inv_transition *d_inverse_neighbors;

	cudaMalloc((void **)&d_prob_matrix, seq_length * num_of_states * max_in_degree * sizeof(double));
	cudaMalloc((void **)&d_last_row_weights, num_of_states * sizeof(double));
	cudaMalloc((void **)&d_inverse_neighbors, num_of_states * max_in_degree * sizeof(inv_transition));

	cudaMemcpy(d_inverse_neighbors, inv_neighbors, num_of_states * max_in_degree *sizeof(inv_transition), cudaMemcpyHostToDevice);

	gpu_forward_matrix(
		states,
		event_sequence,
		d_inverse_neighbors,
		num_of_states,
		max_in_degree,
		d_prob_matrix,
		d_last_row_weights);

	//calculate prefix sums for all weights;
	
	int threads_per_block = 1024;
	int num_of_blocks = std::max(num_of_states * seq_length / threads_per_block, 1);
	prefix_sum<<<num_of_blocks, threads_per_block>>>(d_prob_matrix, max_in_degree);
	cudaDeviceSynchronize();
	/*
	double *prob_matrix = (double *)malloc(seq_length * num_of_states * max_in_degree *sizeof(double));
	cudaMemcpy(prob_matrix, d_prob_matrix, seq_length * num_of_states * max_in_degree * sizeof(double), cudaMemcpyDeviceToHost);
	printf("After prefix sum calculation looks like this:\n");
	print_prob_matrix(prob_matrix, seq_length, num_of_states, max_in_degree);*/

	normalize<<<1,1>>>(d_last_row_weights, num_of_states);
	cudaDeviceSynchronize();

	prefix_sum<<<1,1>>>(d_last_row_weights, num_of_states);
	cudaDeviceSynchronize();


	int *samples = (int *)malloc(seq_length * num_of_samples * sizeof(int));
	for (int i = 0; i < seq_length * num_of_samples; i++) samples[i] = 0;
	int *d_samples;
	cudaMalloc((void **)&d_samples, seq_length * num_of_samples * sizeof(int));
	cudaMemcpy(d_samples, samples, seq_length * num_of_samples * sizeof(int), cudaMemcpyHostToDevice);
	threads_per_block = 1024;
	num_of_blocks = std::max(num_of_samples / threads_per_block, 1);
	backtrack_sample<<<num_of_blocks,threads_per_block>>>(
				d_last_row_weights,
				d_prob_matrix,
				d_inverse_neighbors,
				seq_length,
				num_of_states,
				max_in_degree,
				num_of_samples,
				d_samples,
				rand()
		);
	cudaDeviceSynchronize();
	cudaMemcpy(samples, d_samples, seq_length * num_of_samples * sizeof(int), cudaMemcpyDeviceToHost);
	
	std::vector<std::vector<int> >r;
	for (int i = 0; i < num_of_samples; i++){
		std::vector<int>temp(samples + i * seq_length, samples + (i+1) * seq_length);
		r.push_back(temp);
	}

	cudaFree(d_inverse_neighbors);
	cudaFree(d_prob_matrix);
	cudaFree(d_last_row_weights);
	cudaFree(d_samples);
	free(inv_neighbors);
	free(samples);

	cudaEventRecord(start_sampling);
	cudaEventRecord(stop_sampling);
	cudaEventSynchronize(stop_sampling);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_sampling, stop_sampling);
	cudaEventDestroy(start_sampling);
	cudaEventDestroy(stop_sampling);
	return r;
}