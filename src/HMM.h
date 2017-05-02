#ifndef HMM_H
#define HMM_H

#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>

#include "State.h"
#include "LogNum.h"

template < typename val >
using Matrix = std::vector<std::vector<val> >;

class HMM{
private:
	int kmer_size;
	std::set<char>bases;
	std::vector<State>states;
	std::vector<std::vector<LogNum> >transitions;
	Matrix<LogNum>inverse_transitions;
	std::map<std::string, int> kmer_to_state;
	std::vector<std::string> split_string(std::string &s, char delimiter);
	std::vector<std::string> generate_suffixes();
	std::vector<char> translate_to_bases(std::vector<int>state_sequence);

public:
	void loadModelParams(std::string filename);
	void compute_transitions(double prob_skip, double prob_stay);
	Matrix<LogNum> compute_forward_matrix(std::vector<double> event_sequence);
	std::vector<int> compute_viterbi_path(std::vector<double> event_sequence);
	Matrix<int> generate_samples(int num_of_samples, Matrix<LogNum>&forward_matrix);

};

#endif