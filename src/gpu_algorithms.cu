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
__device__ double log_div(double a, double b){
	if (a != -INFINITY) return a - b;
	return a;
}

__host__
__device__ bool log_greater(double a, double b){
	if (a != -INFINITY && b == -INFINITY) return true;
	if (b == -INFINITY) return true;
	return a > b;
}

__host__
__device__ double log_sum(double a, double b){
	double res = a;
	if (a == -INFINITY){
		res = b;
	}
	else{
		if (a > b){
			res += log1p(exp(b - a));
		}
		else{
			res += log1p(exp(a - b));
		}
	}
	return res;
}

__device__ void traceback_states(int seq_length,
								int num_of_states,
								viterbi_entry *d_viterbi_matrix,
								int *d_state_seq)
{
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
								int *state_seq)
{
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

__global__ void calculate_fw_probs(
		double *d_fw_matrix,
		double *d_prob_matrix,
		inv_transition *d_inverse_neighbors,
		state_params *d_states,
		int num_of_states,
		int num_of_neighbors,
		int i,
		double emission)
{
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	if (l < num_of_states){
		double em_prob = log(emission_probability(d_states[l].mean, d_states[l].stdv, emission));
		double sum = -INFINITY;
		for (int j = 0; j < num_of_neighbors; j++){
			inv_transition t = d_inverse_neighbors[l*num_of_neighbors + j];
			int k = t.state;
			double gen_prob = log_mult(d_fw_matrix[(i-1) * num_of_states + k], t.prob);
			sum = log_sum(sum, log_mult(gen_prob, em_prob));
			d_prob_matrix[i * num_of_states * num_of_neighbors
				+ l * num_of_neighbors + j] = log_mult(gen_prob, em_prob);
		}
		d_fw_matrix[i * num_of_states + l] = sum;
		if (sum != -INFINITY){
			for (int j = 0; j < num_of_neighbors; j++){
				int sh = i * num_of_states * num_of_neighbors + l * num_of_neighbors + j;
				d_prob_matrix[sh] = log_div(d_prob_matrix[sh], sum);
				d_prob_matrix[sh] = exp(d_prob_matrix[sh]);
			}
		}
	}
}

__global__ void normalize(double *d_array, int len){
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	double sum = 0.0;
	for (int i = 0; i < len; i++) sum += d_array[l * len + i];
	for (int i = 0; i < len; i++) d_array[l * len + i] /= sum;
}

__global__ void prefix_sum(double *d_prob_weights, int num_of_neighbors){
	double sum = 0;
	for (int i = 0; i < num_of_neighbors; i++){
		sum += d_prob_weights[i];
		d_prob_weights[i] = sum;
	}
}

void gpu_forward_matrix(
		std::vector<State> &states,
		std::vector<double> &event_sequence,
		inv_transition *d_inverse_neighbors,
		int num_of_states,
		int num_of_neighbors,
		double *d_prob_matrix,
		double *d_last_row_weights)
{

	int seq_length = event_sequence.size();
	LogNum init_transition_prob = LogNum(1.0/(double)num_of_states);

	double *fw_matrix = (double *)malloc(num_of_states * seq_length * sizeof(double));
	double *prob_matrix = (double *)malloc(num_of_states * num_of_neighbors * seq_length * sizeof(double));
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
		for (int j = 0; j < num_of_neighbors; j++){
			prob_matrix[l * num_of_neighbors + j] = -INFINITY;
		}
	}

	/*
	for (int i = 1; i < seq_length; i++){
		for (int l = 0; l < num_of_states; l++){
			cpu_calculate_fw_probs(
				fw_matrix,
				prob_matrix,
				inv_neighbors,
				state_p,
				num_of_states,
				num_of_neighbors,
				i,
				event_sequence[i],
				l);
		}
	}*/

	state_params *d_states;
	double *d_fw_matrix;
	cudaMalloc((void **)&d_fw_matrix, seq_length * num_of_states * sizeof(double));
	cudaMalloc((void **)&d_states, num_of_states * sizeof(state_params));

	cudaMemcpy(d_fw_matrix, fw_matrix, num_of_states * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prob_matrix, prob_matrix, num_of_states * num_of_neighbors* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_states, state_p, num_of_states * sizeof(state_params), cudaMemcpyHostToDevice);

	int threads_per_block = 512;
	int num_of_blocks = std::max(num_of_states / threads_per_block, 1);
	for (int i = 1; i < seq_length; i++){
		calculate_fw_probs<<<num_of_blocks, threads_per_block>>>(
				d_fw_matrix,
				d_prob_matrix,
				d_inverse_neighbors,
				d_states,
				num_of_states,
				num_of_neighbors,
				i,
				event_sequence[i]
			);
		cudaDeviceSynchronize();
	}
	//normalize the prob weights again
	threads_per_block = 1024;
	num_of_blocks = std::max(num_of_states * seq_length / threads_per_block, 1);
	for (int i = 0; i < seq_length * num_of_states; i++){
		normalize<<<num_of_blocks, threads_per_block>>>(d_prob_matrix, num_of_neighbors);
	}
	cudaDeviceSynchronize();
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


	
	cudaMemcpy(fw_matrix, d_fw_matrix, num_of_states * seq_length* sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int j = 0; j < num_of_states; j++){
			printf("%.4f ", fw_matrix[i * num_of_states + j]);
		}
		printf("\n");
	}
	/*
	cudaMemcpy(prob_matrix, d_prob_matrix, num_of_states * seq_length * num_of_neighbors * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int l = 0; l < num_of_states; l++){
			double sum = -INFINITY;
			for (int j = 0; j < num_of_neighbors; j++){
				sum = log_sum(sum, prob_matrix[i * num_of_states * num_of_neighbors
					+ l * num_of_neighbors + j]);
			}
			printf("%.4f ", exp(sum));
		}
		printf("\n");
	}*/

	cudaFree(d_fw_matrix);
	cudaFree(d_states);

	free(fw_matrix);
	free(prob_matrix);
	free(state_p);
	free(last_row);
}

__device__ int discrete_dist(double *weights, int len, curandState *s){
	double val = curand_uniform(s);
	for (int i = 0; i < len; i++){
		if (val <= weights[i]){
			return i;
		}
	}
	return len -1;
}

__global__ void backtrack_sample(
								double *d_last_row_weights,
								double *d_prob_weights,
								inv_transition *d_inv_neighbors,
								int seq_length,
								int num_of_states,
								int num_of_neighbors,
								int num_of_samples,
								int *sample)
{
	unsigned int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (sample_id < num_of_samples){

		unsigned int seed = sample_id;
		curandState s;
		curand_init(seed, 0, 0, &s);

		int curr_state = discrete_dist(d_last_row_weights, num_of_states, &s);
		int i = 0;
		int rem_length = seq_length - 1;
		while (rem_length >= 0){
			sample[sample_id * seq_length + i] = curr_state;
			i++;
			int shift = rem_length * num_of_states * num_of_neighbors + curr_state * num_of_neighbors;
			int next_state_id = discrete_dist(d_prob_weights + shift, num_of_neighbors, &s);
			curr_state = d_inv_neighbors[curr_state * num_of_neighbors + next_state_id].state;
			rem_length--;
		}
	}

}


std::vector<std::vector<int> > gpu_samples(
	int num_of_samples,
	std::vector<State> &states,
	std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
	std::vector<double>&event_sequence){

	int num_of_states = states.size();
	int seq_length = event_sequence.size();
	int num_of_neighbors = inverse_neighbors[0].size();

	//convert transitions to struct type
	inv_transition *inv_neighbors = (inv_transition *)malloc(num_of_states * num_of_neighbors *sizeof(inv_transition));
	for (int i = 0; i < num_of_states; i++){
		for (int j = 0; j < inverse_neighbors[i].size(); j++){
			inv_transition t;
			t.state = inverse_neighbors[i][j].first;
			t.prob = (inverse_neighbors[i][j].second).exponent;
			inv_neighbors[i * num_of_neighbors + j] = t;
		}
	}

	double *d_prob_matrix;
	double *d_last_row_weights;
	inv_transition *d_inverse_neighbors;

	cudaMalloc((void **)&d_prob_matrix, seq_length * num_of_states * num_of_neighbors * sizeof(double));
	cudaMalloc((void **)&d_last_row_weights, num_of_states * sizeof(double));
	cudaMalloc((void **)&d_inverse_neighbors, num_of_states * num_of_neighbors * sizeof(inv_transition));

	cudaMemcpy(d_inverse_neighbors, inv_neighbors, num_of_states * num_of_neighbors *sizeof(inv_transition), cudaMemcpyHostToDevice);

	gpu_forward_matrix(
		states,
		event_sequence,
		d_inverse_neighbors,
		num_of_states,
		num_of_neighbors,
		d_prob_matrix,
		d_last_row_weights);

	//DEBUG SECTION
	/*
	printf("DEBUG\n");
	double *prob_matrix = (double *)malloc(seq_length * num_of_states * num_of_neighbors * sizeof(double));
	cudaMemcpy(prob_matrix, d_prob_matrix, seq_length * num_of_states * num_of_neighbors * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int st = 0; st < num_of_states; st++){
			printf("[");
			for (int ne = 0; ne < num_of_neighbors; ne++){
				printf("%f,", prob_matrix[i*num_of_states*num_of_neighbors + st*num_of_neighbors + ne]);
			}
			printf("],");
		}
		printf("\n");
	}

	
	//DEBUG_SECTION*/

	//calculate prefix sums for all weights;
	
	int threads_per_block = 512;
	int num_of_blocks = std::max(num_of_states / threads_per_block, 1);
	for (int i = 0; i < seq_length * num_of_states; i++){
		prefix_sum<<<num_of_blocks, threads_per_block>>>(d_prob_matrix + i * num_of_neighbors, num_of_neighbors);
	}
	/*
	cudaMemcpy(prob_matrix, d_prob_matrix, seq_length * num_of_states * num_of_neighbors * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int st = 0; st < num_of_states; st++){
			printf("[");
			for (int ne = 0; ne < num_of_neighbors; ne++){
				printf("%f,", prob_matrix[i*num_of_states*num_of_neighbors + st*num_of_neighbors + ne]);
			}
			printf("],");
		}
		printf("\n");
	}
	free(prob_matrix);*/

	normalize<<<1,1>>>(d_last_row_weights, num_of_states);
	cudaDeviceSynchronize();
	prefix_sum<<<1,1>>>(d_last_row_weights, num_of_states);
	cudaDeviceSynchronize();


	int *samples = (int *)malloc(seq_length * num_of_samples * sizeof(int));
	for (int i = 0; i < seq_length * num_of_samples; i++) samples[i] = 0;
	int *d_samples;
	cudaMalloc((void **)&d_samples, seq_length * num_of_samples * sizeof(int));
	cudaMemcpy(d_samples, samples, seq_length * num_of_samples * sizeof(int), cudaMemcpyHostToDevice);
	threads_per_block = 512;
	num_of_blocks = std::max(num_of_samples / threads_per_block, 1);
	backtrack_sample<<<num_of_blocks,threads_per_block>>>(
				d_last_row_weights,
				d_prob_matrix,
				d_inverse_neighbors,
				seq_length,
				num_of_states,
				num_of_neighbors,
				num_of_samples,
				d_samples
		);
	cudaDeviceSynchronize();
	cudaMemcpy(samples, d_samples, seq_length * num_of_samples * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num_of_samples; i++){
		for (int j = 0; j < seq_length; j++){
			printf("%d ", samples[i * seq_length + j]);
		}
		printf("\n\n");
	}



	cudaFree(d_inverse_neighbors);
	cudaFree(d_prob_matrix);
	cudaFree(d_last_row_weights);
	cudaFree(d_samples);
	free(inv_neighbors);
	free(samples);
	std::vector<std::vector<int> >res;
	return res;
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
	int num_of_neighbors = inverse_neighbors[0].size();

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
	inv_transition *inv_neighbors = (inv_transition *)malloc(num_of_states * num_of_neighbors *sizeof(inv_transition));
	for (int i = 0; i < num_of_states; i++){
		for (int j = 0; j < inverse_neighbors[i].size(); j++){
			inv_transition t;
			t.state = inverse_neighbors[i][j].first;
			t.prob = (inverse_neighbors[i][j].second).exponent;
			inv_neighbors[i * num_of_neighbors + j] = t;
		}
	}

	viterbi_entry *d_viterbi_matrix;
	inv_transition *d_inverse_neighbors;
	state_params *d_states;
	int *d_state_seq;

	cudaMalloc((void **)&d_viterbi_matrix, seq_length * num_of_states * sizeof(viterbi_entry));
	cudaMalloc((void **)&d_inverse_neighbors, num_of_states * num_of_neighbors * sizeof(inv_transition));
	cudaMalloc((void **)&d_states, num_of_states * sizeof(state_params));

	cudaMemcpy(d_viterbi_matrix, viterbi_matrix_row, num_of_states * sizeof(viterbi_entry), cudaMemcpyHostToDevice);
	cudaMemcpy(d_states, state_p, num_of_states * sizeof(state_params), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inverse_neighbors, inv_neighbors, num_of_states * num_of_neighbors *sizeof(inv_transition), cudaMemcpyHostToDevice);

	int threads_per_block = 512;
	int num_of_blocks = std::max(num_of_states / threads_per_block, 1);

	for (int i = 1; i < seq_length; i++){
		calculate_viterbi_prob<<<num_of_blocks, threads_per_block>>>(
			d_viterbi_matrix,
			d_inverse_neighbors,
			d_states,
			num_of_states,
			num_of_neighbors,
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

__global__ void discrete_dist_brute(double *weights, int len, int *res){
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seed = id;
	curandState s;
	curand_init(seed, 0, 0, &s);
	double val = curand_uniform(&s);
	for (int i = 0; i < len; i++){
		if (val <= weights[i]){
			res[id] = i;
			return;
		}
	}
	res[id] = len -1;
}

__global__ void discrete_dist_bins(double *weights, int len, double val, int *res){
}

__global__ void rand_nums(double *nums){
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < 32){
		unsigned int seed = i;
		curandState s;
		curand_init(seed, 0, 0, &s);
		nums[i] = curand_uniform(&s);
	}
}

void test_random_stuff(){

	double a = -INFINITY;
	double b = 2.0;
	printf("%f", exp(log_sum(a,b)));

	/*
	double *nums = (double *)malloc(32 * sizeof(double));
	double *d_nums;
	cudaMalloc((void **)&d_nums, 32 * sizeof(double));
	rand_nums<<<4,8>>>(d_nums);
	cudaDeviceSynchronize();
	cudaMemcpy(nums, d_nums, 32 * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 32; i++){
		printf("%.5f ", nums[i]);
	}
	printf("\n");*/

	double *weights = (double *)malloc(5 * sizeof(double));
	weights[0] = 0.2;
	weights[1] = 0.35;
	weights[2] = 0.7;
	weights[3] = 0.8;
	weights[4] = 1.0;

	double *d_weights;
	int *d_res;
	int num_of_rands = 1048576;
	cudaMalloc((void **)&d_res, num_of_rands * sizeof(int));
	cudaMalloc((void **)&d_weights, 5 * sizeof(double));
	cudaMemcpy(d_weights, weights, 5 * sizeof(double), cudaMemcpyHostToDevice);
	discrete_dist_brute<<<1024,1024>>>(d_weights, 5, d_res);
	cudaDeviceSynchronize();
	int *res = (int *)malloc(num_of_rands * sizeof(int));
	cudaMemcpy(res, d_res, num_of_rands * sizeof(int), cudaMemcpyDeviceToHost);
	int *counts = (int *)malloc(5 * sizeof(int));
	for (int i = 0; i < 5; i++) counts[i] = 0;
	for (int i = 0; i < num_of_rands; i++){
		counts[res[i]]++;
	}
	for (int i = 0; i < 5; i++){
		printf("%d generated %d times\n", i, counts[i]);
	}
	cudaFree(d_res);
	cudaFree(d_weights);
	free(weights);
	free(counts);
	free(res);

	//free(nums);
	//cudaFree(d_nums);
}