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
	int num_of_states = 0;
	int max_skip = 2;

	std::set<std::string>bases;
	std::vector<State>states;
	Matrix<LogNum>transitions;
	Matrix<std::pair<int, LogNum> >inverse_neighbors;
	LogNum init_transition_prob = LogNum(0.0);

	std::map<std::string, int> kmer_to_state;

	double scale = 1;
	double shift = 0;
	double prob_skip = 0.3;
	double prob_stay = 0.1;

	std::vector<std::string> split_string(std::string &s, char delimiter);
	int kmer_overlap(std::string from, std::string to);
	std::vector<std::vector<int> > cpu_samples(int num_of_samples, std::vector<double> &event_sequence, int seed);
	std::vector<std::vector<std::string> > generate_subkmers();

public:
	void load_model_params(std::string filename);
	void compute_transitions();
	Matrix<std::vector<double> > compute_forward_matrix(std::vector<double>& event_sequence);

	std::vector<int> compute_viterbi_path(std::vector<double>&event_sequence, std::string method);

	std::vector<int> cpu_viterbi_path(std::vector<double>&event_sequence);

	std::vector<int> backtrack_sample(int last_state, int l, Matrix<std::vector<double> > &prob_weights, std::default_random_engine gen);

	std::vector<std::vector<int> > generate_samples(int num_of_samples, std::vector<double>&event_sequence, std::string method);

	std::vector<char> translate_to_bases(int *state_sequence, int seq_length) const;

	std::vector<char> translate_to_bases(std::vector<int> state_sequence) const;

	void adjust_scaling(std::vector<double>& event_sequence);
	
	void set_stay_prob(double prob);
	void set_skip_prob(double prob);

	void gpu_viterbi(std::vector<double>& event_sequence);

	void dump_emissions(std::vector<double>&event_sequence);
};

#endif