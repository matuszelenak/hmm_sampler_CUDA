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
	int max_in_degree = 0;

	int max_skip = 2;

	std::set<std::string>bases;
	std::vector<State>states;
	Matrix<std::pair<int, LogNum> >inverse_neighbors;
	std::vector<std::string> kmers;
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
	void set_stay_prob(double prob);
	void set_skip_prob(double prob);
	void set_max_skip(int skip);
	void adjust_scaling(std::vector<double>& event_sequence);

	std::pair<Matrix<std::vector<LogNum> >, Matrix<LogNum> > compute_forward_matrix(std::vector<double>& event_sequence);

	std::vector<int> compute_viterbi_path(std::vector<double>&event_sequence, std::string method);

	std::vector<int> cpu_viterbi_path(std::vector<double>&event_sequence);

	std::vector<int> backtrack_sample(std::vector<LogNum>&last_row_weights, Matrix<std::vector<LogNum> > &prob_weights, int seq_length);

	std::vector<std::vector<int> > generate_samples(int num_of_samples, std::vector<double>&event_sequence, std::string method, int version);

	Matrix<LogNum> compute_forward_matrix_v2(std::vector<double>& event_sequence);

	std::vector<int> backtrack_sample_v2(Matrix<LogNum>&fw_matrix, std::vector<double>&event_sequence);

	std::vector<std::vector<int> > cpu_samples_v2(int num_of_samples, std::vector<double>&event_sequence, int seed);

	std::vector<char> cpu_decode_path(std::vector<int> &state_sequence) const;

	std::vector<std::vector<char> > cpu_decode_paths(std::vector<std::vector<int> >&samples);

	std::vector<std::vector<char> > decode_paths(std::vector<std::vector<int> >&samples, std::string method);
};

#endif