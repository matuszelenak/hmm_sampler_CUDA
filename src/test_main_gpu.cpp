#include "gpu_algorithms.h"
#include "State.h"
#include "LogNum.h"
#include <vector>
#include "HMM.h"

int main(int argc, char const *argv[])
{
	HMM hmm;
	hmm.load_model_params("test_data/simple.hmm");
	hmm.compute_transitions();
	
	std::vector<double>event_seq = {3,15,7,2,6,9,7};
	hmm.compute_viterbi_path(event_seq, "GPU");
	hmm.compute_viterbi_path(event_seq, "CPU");
	return 0;
}