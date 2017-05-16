

__global__ void kernel(double *arr, int len){
	double prev = 0.0;
	for (int i = 0; i < len; i++){
		if (i % 2){
			arr[i] = prev - i;
		}
		else{
			arr[i] = prev + i;
		}
		prev = arr[i];
	}
}

int launch_kernel(int n){
	double *d_arr;

	cudaEvent_t start_fwm, stop_fwm;
	cudaEventCreate(&start_fwm);
	cudaEventCreate(&stop_fwm);
	cudaEventRecord(start_fwm);

	cudaMalloc((void **)&d_arr, n * sizeof(double));

	kernel<<<1,1>>>(d_arr, n);
	cudaDeviceSynchronize();

	cudaEventRecord(stop_fwm);
	cudaEventSynchronize(stop_fwm);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_fwm, stop_fwm);
	cudaEventDestroy(start_fwm);
	cudaEventDestroy(stop_fwm);

	cudaFree(d_arr);

	return (int)round(milliseconds);
}
