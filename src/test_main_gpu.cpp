#include "gpu_algorithms.h"
#include "State.h"
#include "LogNum.h"
#include <vector>
#include "HMM.h"
#include <chrono>
#include <iostream>

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

int main(int argc, char const *argv[])
{
	HMM hmm;
	hmm.load_model_params("test_data/simple.hmm");
	hmm.compute_transitions();
	//hmm.set_skip_prob(0.0);
	
	std::vector<double>event_seq = {3,15,7,2,6,9,7,4,7,5,1,2,4,8,1,6,8,1,5,3,4,8,3,1,12,7,14,10,15,9,4,6,7,3,8,14};
	auto r = hmm.compute_forward_matrix(event_seq);
	auto start = system_clock::now();
	//test_random_stuff();
	hmm.gpu_sample(100,event_seq);
	std::cout << "Generating samples took " << duration_cast<milliseconds>(system_clock::now() - start).count() << "ms\n";
	//hmm.dump_emissions(event_seq);
	//std::vector<std::vector<std::vector<double> > > k = hmm.compute_forward_matrix(event_seq);
	//hmm.compute_viterbi_path(event_seq, "GPU");
	//hmm.compute_viterbi_path(event_seq, "CPU");

	return 0;
}