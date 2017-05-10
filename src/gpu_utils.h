#ifndef GPU_UTILS_H
#define GPU_UTILS_H

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

__host__
__device__ bool log_less(double a, double b);

__host__
__device__ double emission_probability(double mean, double stdv, double emission);

__host__
__device__ double log_mult(double a, double b);

__host__
__device__ double log_div(double a, double b);

__host__
__device__ bool log_greater(double a, double b);

__host__
__device__ double log_sum(double a, double b);

#endif