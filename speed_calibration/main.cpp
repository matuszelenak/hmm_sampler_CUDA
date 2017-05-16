#include "calibration.h"
#include <vector>
#include <chrono>
#include <cstdio>

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

long launch_cpu(int n){
	auto start = system_clock::now();
	std::vector<double>arr(n);

	
	double prev = 0.0;
	for (int i = 0; i < n; i++){
		if (i % 2){
			arr[i] = prev - i;
		}
		else{
			arr[i] = prev + i;
		}
		prev = arr[i];
	}
	return duration_cast<milliseconds>(system_clock::now() - start).count();
}

int main(int argc, char const *argv[])
{
	int n = atoi(argv[1]);
	int gpu_res = launch_kernel(n);
	long cpu_res = launch_cpu(n);
	printf("CPU %ld ms\n", cpu_res);
	printf("GPU %d ms\n", gpu_res);
	return 0;
}