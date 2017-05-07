#ifndef GPU_ALGORITHMS_H
#define GPU_ALGORITHMS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include "State.h"
#include "LogNum.h"

struct inv_transition {
	int state;
	double prob;
};

struct state_params {
	double mean;
	double stdv;
};

struct viterbi_entry{
	double prob;
	int state;
};

__global__ void calculate_fw_probs();

__global__ void calculate_viterbi_prob(
							LogNum *d_viterbi_matrix,
							inv_transition *d_inverse_neighbors,
							state_params *d_states,
							int num_of_states,
							int num_of_neighbors,
							int i,
							double emission);

void gpu_forward_matrix();

std::vector<int> gpu_viterbi_path(std::vector<State> &states,
		std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
		std::vector<double> &event_sequence);

void gpu_forward_matrix(
		std::vector<State> &states,
		std::vector<double> &event_sequence,
		inv_transition *d_inverse_neighbors,
		int num_of_states,
		int num_of_neighbors,
		double *d_prob_matrix,
		double *d_last_row_weights);

std::vector<std::vector<int> > gpu_samples(
	int num_of_samples,
	std::vector<State> &states,
	std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
	std::vector<double>&event_sequence);

void test_random_stuff();

#endif