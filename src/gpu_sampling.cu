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
	int l = threadIdx.x + blockIdx.x * blockDim.x;
	double sum = 0;
	for (int i = 0; i < num_of_neighbors; i++){
		sum += d_prob_weights[l * num_of_neighbors + i];
		d_prob_weights[l * num_of_neighbors] = sum;
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
	normalize<<<num_of_blocks, threads_per_block>>>(d_prob_matrix, num_of_neighbors);
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


	/*
	cudaMemcpy(fw_matrix, d_fw_matrix, num_of_states * seq_length* sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < seq_length; i++){
		for (int j = 0; j < num_of_states; j++){
			printf("%.4f ", fw_matrix[i * num_of_states + j]);
		}
		printf("\n");
	}
	
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
		int i = seq_length - 1;
		int rem_length = seq_length - 1;
		while (rem_length >= 0){
			sample[sample_id * seq_length + i] = curr_state;
			i--;
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
	int num_of_blocks = std::max(num_of_states * seq_length / threads_per_block, 1);
	prefix_sum<<<num_of_blocks, threads_per_block>>>(d_prob_matrix, num_of_neighbors);
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
	/*
	for (int i = 0; i < num_of_samples; i++){
		for (int j = 0; j < seq_length; j++){
			printf("%d ", samples[i * seq_length + j]);
		}
		printf("\n\n");
	}*/
	std::vector<std::vector<int> >r;
	for (int i = 0; i < num_of_samples; i++){
		std::vector<int>temp(samples + i * seq_length, samples + (i+1) * seq_length);
		r.push_back(temp);
	}
	/*
	for (int i = 0; i < r.size(); i++){
		for (int j = 0; j < r[i].size(); j++){
			printf("%d->", r[i][j]);
		}
		printf("\n");
	}*/

	printf("Copying succesful\n");

	cudaFree(d_inverse_neighbors);
	cudaFree(d_prob_matrix);
	cudaFree(d_last_row_weights);
	cudaFree(d_samples);
	free(inv_neighbors);
	free(samples);
	return r;
}