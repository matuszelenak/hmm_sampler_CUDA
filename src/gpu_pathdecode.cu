#include "gpu_algorithms.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

__device__ bool cmp(char *a, char *b, int len){
	for (int i = 0; i < len; i++){
		if (a[i] != b[i]) return false;
	}
	return true;
}

__global__ void decode_block(int *d_samples,
							char *d_decoded,
							char *d_kmers,
							int kmer_size,
							int sample_length,
							int padded_sample_length,
							int block_size,
							int threads_per_sample,
							int num_of_samples)
{
	unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_id * block_size >= num_of_samples * padded_sample_length)
		return;

	unsigned int offset = thread_id * block_size;
	int init_index;
	int prev_state;
	if (thread_id % threads_per_sample == 0){
		int first_state = d_samples[offset];
		for (int i = 0; i < kmer_size; i++){
			d_decoded[offset * kmer_size + i] = d_kmers[first_state * kmer_size + i];
		}
		init_index = 1;
		prev_state = d_samples[offset];
	}
	else{
		init_index = 0;
		prev_state = d_samples[offset - 1];
	}
	
	for (int i = init_index; i < block_size; i++){
		if (((offset + i) % padded_sample_length) == sample_length){
			break;
		}
		
		int curr_state = d_samples[offset + i];
		for (int prefix = 0; prefix <= kmer_size; prefix++){
			char * prev_suffix = d_kmers + prev_state * kmer_size + prefix;
			char * curr_prefix = d_kmers + curr_state * kmer_size;
			if (cmp(prev_suffix, curr_prefix, kmer_size - prefix)){
				char * appendage = d_kmers + curr_state * kmer_size + (kmer_size - prefix);
				for (int k = 0; k < prefix; k++){
					d_decoded[offset * kmer_size + i * kmer_size + k] = appendage[k];
				}
				break;
			}
		}
		prev_state = curr_state;
	}
}

std::vector<std::vector<char> > gpu_decode_paths(std::vector<std::vector<int> >&samples, std::vector<std::string>&kmers){
	int num_of_samples = samples.size();
	int sample_length = samples[0].size();
	int num_of_states = kmers.size();
	int kmer_size = kmers[0].size();

	int num_of_threads = 16384;
	int threads_per_sample = min(max((int)floor((double)num_of_threads / (double)num_of_samples), 1), sample_length);
	int block_per_thread = (int)ceil((double)sample_length / (double)threads_per_sample);
	int padded_sample_length = threads_per_sample * block_per_thread;

	int *d_samples;
	cudaMalloc((void **)&d_samples, num_of_samples * padded_sample_length * sizeof(int));
	for (int i = 0; i < num_of_samples; i++){
		cudaMemcpy(d_samples + i* threads_per_sample * block_per_thread, &samples[i][0], sample_length * sizeof(int), cudaMemcpyHostToDevice);
	}

	char *d_decoded;
	cudaMalloc((void **)&d_decoded, num_of_samples * padded_sample_length * kmer_size * sizeof(char));

	char *d_kmers;
	cudaMalloc((void **)&d_kmers, num_of_states * kmer_size * sizeof(char));
	for (int i = 0; i < kmers.size(); i++){
		cudaMemcpy(d_kmers + i * kmer_size, &kmers[i][0], kmer_size * sizeof(char), cudaMemcpyHostToDevice);
	}

	cudaEvent_t start_decode, stop_decode;
	cudaEventCreate(&start_decode);
	cudaEventCreate(&stop_decode);
	cudaEventRecord(start_decode);

	int threads_per_block = 1024;
	int num_of_blocks = max((num_of_threads / threads_per_block),1);
	decode_block<<<num_of_blocks, threads_per_block>>>(
							d_samples,
							d_decoded,
							d_kmers,
							kmer_size,
							sample_length,
							padded_sample_length,
							block_per_thread,
							threads_per_sample,
							num_of_samples);
	cudaDeviceSynchronize();

	cudaEventRecord(stop_decode);
	cudaEventSynchronize(stop_decode);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_decode, stop_decode);
	printf("DECODE GPU TOOK %d ms\n", (int)round(milliseconds));
	cudaEventDestroy(start_decode);
	cudaEventDestroy(stop_decode);

	char *decoded = (char*)malloc(num_of_samples * padded_sample_length * kmer_size * sizeof(char));
	cudaMemcpy(decoded, d_decoded, num_of_samples * padded_sample_length * kmer_size * sizeof(char), cudaMemcpyDeviceToHost);

	std::vector<std::vector<char> > r(num_of_samples);
	for (int i = 0; i < num_of_samples; i++){
		for (int j = 0; j < padded_sample_length * kmer_size; j++){
			int offset = i * padded_sample_length * kmer_size + j;
			if (decoded[offset] == 0) continue;
			r[i].push_back(decoded[offset]);
		}
	}

	cudaFree(d_samples);
	cudaFree(d_kmers);
	cudaFree(d_decoded);
	free(decoded);

	return r;
}