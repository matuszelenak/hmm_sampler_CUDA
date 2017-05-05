#include "gpu_algorithms.h"

__global__ void calculate_fw_probs()
{

}

void gpu_forward_matrix(){
	calculate_fw_probs<<<1, 1>>>();
}