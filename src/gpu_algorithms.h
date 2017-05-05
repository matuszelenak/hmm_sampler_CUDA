#ifndef GPU_ALGORITHMS_H
#define GPU_ALGORITHMS_H

#include <cuda_runtime.h>
#include <cuda.h>

__global__ void calculate_fw_probs();

void gpu_forward_matrix();

#endif