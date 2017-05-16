#include "gpu_algorithms.h"
#include "gpu_utils.h"
#include <cstdio>
#include <algorithm>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void calculate_fw_probs(
		double *d_fw_matrix,
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
		}
		d_fw_matrix[i * num_of_states + l] = sum;

	}
}

__host__
__device__ void normalize(double *d_array, int len){
	double sum = -INFINITY;
	for (int i = 0; i < len; i++){
		if (d_array[i] == INFINITY) break;
		sum = log_sum(d_array[i], sum);
	} 
	for (int i = 0; i < len; i++){
		if (d_array[i] == INFINITY) break;
		d_array[i] = log_div(d_array[i], sum);
	}
}
__host__
__device__ void prefix_sum(double *d_prob_weights, int max_in_degree){
	double sum = -INFINITY;
	for (int i = 0; i < max_in_degree; i++){
		if (d_prob_weights[i] == INFINITY) break;
		sum = log_sum(d_prob_weights[i], sum);
		d_prob_weights[i] = sum;
	}
}

__device__ int discrete_dist_bin(double *weights, int len, curandState *s){
	double val = log(curand_uniform(s));
	int pivot_l, pivot_r;
	double val_l, val_r;
	int left = 0;
	int right = len + 1;
	pivot_r = (left + right) / 2;
	pivot_l = pivot_r - 1;
	while(left != right) {
		if (pivot_r == len) val_r = 0; else val_r = weights[pivot_r];
		if (pivot_l == -1) val_l = -INFINITY; else val_l = weights[pivot_l];
		if (log_less(val, val_l)){
			right = pivot_r;
			pivot_r = (left + right) / 2;
			pivot_l = pivot_r - 1;
		} else
		if (log_less(val_r, val)){
			left = pivot_r;
			pivot_r = (left + right) / 2;
			pivot_l = pivot_r - 1;
		}
		else{
			if (pivot_r == len) return pivot_r - 1;
			return pivot_r;
		}
	}
	if (pivot_r == len) return pivot_r - 1;
	return pivot_r;
}

__device__ int discrete_dist_2(double *weights, int max_in_degree, curandState *s){
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
								inv_transition *d_inv_neighbors,
								double *d_fw_matrix,
								double *d_sequence,
								state_params *d_states,
								double *d_state_weights,
								int seq_length,
								int num_of_states,
								int max_in_degree,
								int num_of_samples,
								int *sample,
								int seed)
{
	unsigned int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_id < num_of_samples){

		double *d_last_row_weights = d_fw_matrix + (seq_length - 1) * num_of_states;
		normalize(d_last_row_weights, num_of_states);
		prefix_sum(d_last_row_weights, num_of_states);
		

		curandState s;
		curand_init(seed, sample_id, 0, &s);

		int curr_state = discrete_dist_2(d_last_row_weights, num_of_states, &s);
		sample[sample_id * seq_length + seq_length - 1] = curr_state;
		int i = seq_length - 2;
		int rem_length = seq_length - 1;

		while (rem_length > 0){
			double em_prob = emission_probability(d_states[curr_state].mean, d_states[curr_state].stdv, d_sequence[rem_length - 1]);
			int actual_neighbors = 0;
			for (int j = 0; j < max_in_degree; j++){
				inv_transition t = d_inv_neighbors[curr_state * max_in_degree + j];
				if (t.state == -1) break;
				
				actual_neighbors++;
				d_state_weights[j] = log_mult(d_fw_matrix[(rem_length - 1) * num_of_states + t.state], 
										log_mult(t.prob,em_prob));
			}
			
			for (int j = actual_neighbors; j < max_in_degree; j++) d_state_weights[j] = INFINITY;

			normalize(d_state_weights, actual_neighbors);
			prefix_sum(d_state_weights, actual_neighbors);

			int next_state_id = discrete_dist_2(d_state_weights, actual_neighbors, &s);
			curr_state = d_inv_neighbors[curr_state * max_in_degree + next_state_id].state;
			sample[sample_id * seq_length + i] = curr_state;
			rem_length--;
			i--;
		}
	}
}

std::vector<std::vector<int> > gpu_samples_v2(
	int num_of_samples,
	std::vector<State> &states,
	std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
	int max_in_degree,
	std::vector<double>&event_sequence)
{
	int num_of_states = states.size();
	int seq_length = event_sequence.size();

	LogNum init_transition_prob = LogNum(1.0/(double)num_of_states);

	double *fw_matrix = (double *)malloc(num_of_states * sizeof(double));
	for (int i = 0; i < num_of_states; i++){
		fw_matrix [i] = (init_transition_prob * states[i].get_emission_probability(event_sequence[0])).exponent;
	}

	state_params *state_p = (state_params *)malloc(num_of_states * sizeof(state_params));
	for (int i = 0; i < num_of_states; i++){
		state_params s;
		s.mean = states[i].corrected_mean;
		s.stdv = states[i].corrected_stdv;
		state_p[i] = s;
	}

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

	double *d_sequence;
	cudaMalloc((void **)&d_sequence, seq_length * sizeof(double));
	cudaMemcpy(d_sequence, &event_sequence[0], seq_length * sizeof(double), cudaMemcpyHostToDevice);

	double *d_fw_matrix;
	cudaMalloc((void **)&d_fw_matrix, seq_length * num_of_states * sizeof(double));
	cudaMemcpy(d_fw_matrix, fw_matrix, num_of_states * sizeof(double), cudaMemcpyHostToDevice);
	
	state_params *d_states;
	cudaMalloc((void **)&d_states, num_of_states * sizeof(state_params));
	cudaMemcpy(d_states, state_p, num_of_states * sizeof(state_params), cudaMemcpyHostToDevice);

	inv_transition *d_inverse_neighbors;
	cudaMalloc((void **)&d_inverse_neighbors, num_of_states * max_in_degree * sizeof(inv_transition));
	cudaMemcpy(d_inverse_neighbors, inv_neighbors, num_of_states * max_in_degree *sizeof(inv_transition), cudaMemcpyHostToDevice);

	double *d_state_weights;
	cudaMalloc((void **)&d_state_weights, max_in_degree * sizeof(double));

	int threads_per_block = 1024;
	int num_of_blocks = std::max((int)ceil((double)(num_of_states) / (double)threads_per_block), 1);

	for (int i = 1; i < seq_length; i++){
		calculate_fw_probs<<<num_of_blocks, threads_per_block>>>(
				d_fw_matrix,
				d_inverse_neighbors,
				d_states,
				num_of_states,
				max_in_degree,
				i,
				event_sequence[i]
			);
		cudaDeviceSynchronize();
	}

	int *d_samples;
	cudaMalloc((void **)&d_samples, seq_length * num_of_samples * sizeof(int));

	threads_per_block = 1024;
	num_of_blocks = std::max((int)ceil((double)(num_of_samples) / (double)threads_per_block), 1);
	backtrack_sample<<<num_of_blocks,threads_per_block>>>(
				d_inverse_neighbors,
				d_fw_matrix,
				d_sequence,
				d_states,
				d_state_weights,
				seq_length,
				num_of_states,
				max_in_degree,
				num_of_samples,
				d_samples,
				rand()
		);
	cudaDeviceSynchronize();

	std::vector<std::vector<int> >r;

	int *samples = (int *)malloc(seq_length * num_of_samples * sizeof(int));
	cudaMemcpy(samples, d_samples, seq_length * num_of_samples * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < num_of_samples; i++){
		std::vector<int>temp(samples + i * seq_length, samples + (i+1) * seq_length);
		r.push_back(temp);
	}

	cudaFree(d_fw_matrix);
	cudaFree(d_sequence);
	cudaFree(d_states);
	cudaFree(d_inverse_neighbors);
	cudaFree(d_samples);
	cudaFree(d_state_weights);
	free(fw_matrix);
	free(inv_neighbors);

	return r;
}