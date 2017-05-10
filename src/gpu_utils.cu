#include "gpu_utils.h"
#include <cstdio>
#include <algorithm>
#include <cmath>

__host__
__device__ bool log_less(double a, double b){
	if (a == -INFINITY && b != -INFINITY) return true;
	if (a == -INFINITY) return false;
	return a < b;
}

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
	if (a == -INFINITY){
		return b;
	}
	else{
		if (a > b){
			return a + log(1 + exp(b - a));
		}
		else{
			return b + log(1 + exp(a - b));
		}
	}
}