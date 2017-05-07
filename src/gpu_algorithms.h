#ifndef GPU_ALGORITHMS_H
#define GPU_ALGORITHMS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include "State.h"
#include "LogNum.h"

std::vector<int> gpu_viterbi_path(std::vector<State> &states,
		std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
		std::vector<double> &event_sequence);

std::vector<std::vector<int> > gpu_samples(
	int num_of_samples,
	std::vector<State> &states,
	std::vector<std::vector<std::pair<int, LogNum> > > &inverse_neighbors,
	std::vector<double>&event_sequence);

#endif