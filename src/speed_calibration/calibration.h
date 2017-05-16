#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void kernel(double * arr, int len);

int launch_kernel(int n);

#endif