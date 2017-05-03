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

	double scale = 1;
	double shift = 0;
	double prob_skip = 0.3;
	double prob_stay = 0.1;

	std::vector<std::string> split_string(std::string &s, char delimiter);
	std::vector<std::string> generate_suffixes();
	int kmer_overlap(std::string from, std::string to);

public:
	void loadModelParams(std::string filename);
	void compute_transitions();
	Matrix<LogNum> compute_forward_matrix(std::vector<double>&event_sequence);
	std::vector<int> compute_viterbi_path(std::vector<double>&event_sequence);
	Matrix<int> generate_samples(int num_of_samples, Matrix<LogNum>&forward_matrix);
	std::vector<char> translate_to_bases(std::vector<int>&state_sequence);
	void adjust_scaling(std::vector<double>& event_sequence);
	void set_stay_prob(double prob);
	void set_skip_prob(double prob);
};

#endif