#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include <cstdio>

#include "State.h"
#include "LogNum.h"
#include "HMM.h"

template < typename val >
using Matrix = std::vector<std::vector<val> >;

std::vector<std::string>HMM::generate_suffixes(){
	std::vector<std::string>res;
	for (std::set<char>::iterator it = bases.begin(); it != bases.end(); ++it){
		std::string acc;
		acc.push_back(*it);
		for (std::set<char>::iterator it2 = bases.begin(); it2 != bases.end(); ++it2){
			acc.push_back(*it2);
			res.push_back(acc);
		}
	}
	return res;
}

std::vector<std::string>HMM::split_string(std::string &s, char delimiter){
	std::vector<std::string> res;
	std::string acc = "";
	for (int i = 0; i < s.size(); i++){
		if ((s[i] == delimiter) && (acc.size() > 0)){
			res.push_back(acc);
			acc = "";
		}
		if (s[i] != delimiter) acc += s[i];
	}
	return res;
}

void HMM::loadModelParams(std::string filename){
	states.clear();
	State init_state;
	init_state.setParams(0,0,true);
	init_state.kmer_label = std::string(kmer_size, ' ');
	states.push_back(init_state);

	std::ifstream model_file;
	model_file.open(filename);
	std::string line;
	std::string kmer_label;
	while(getline(model_file, line)) {
		std::vector<std::string> line_elements = split_string(line, '\t');
		kmer_label = line_elements[0];
		double mean = stod(line_elements[1], 0);
		double stdv = stod(line_elements[2], 0);
		//TODO figure out what the remaining two parameters in Nanocall models are for
		State state;
		state.kmer_label = kmer_label;
		std::cout << kmer_label << std::endl;
		state.setParams(mean, stdv, false);
		kmer_to_state.insert(std::pair<std::string, int>(kmer_label, states.size()));
		states.push_back(state);
		for (int i = 0; i < kmer_label.size(); i++){
			bases.insert(kmer_label[i]);
		}
	kmer_size = kmer_label.size();
	}
	model_file.close();
}

void HMM::compute_transitions(double prob_skip, double prob_stay){
	transitions.clear();

	inverse_transitions.clear();
	//initialize all probabilites to zero
	for (int i = 0; i < states.size(); i++){
		std::vector<LogNum>t(states.size(), LogNum(0.0));
		transitions.push_back(t);
		std::vector<LogNum>t2(states.size(), LogNum(0.0));
		inverse_transitions.push_back(t2);
	}
	//initialize transitions from the init state to all others
	for (int i = 1; i < states.size(); i++){
		transitions[0][i] = LogNum(1/(states.size() - 1));
		inverse_transitions[i][0] = LogNum(1/(states.size() - 1));
	}
	//cyclic transitions to states
	for (int i = 1; i < states.size(); i++){
		transitions[i][i] = LogNum(prob_stay);
		inverse_transitions[i][i] = LogNum(prob_stay);
	}
	//transitions that skip base
	std::vector<std::string> suffixes = generate_suffixes();
	for (std::map<std::string, int>::iterator it = kmer_to_state.begin(); it != kmer_to_state.end(); ++it){
		std::string kmer_from = it -> first;
		int state_from = it -> second;
		std::string prefix = kmer_from.substr(2,kmer_size - 2);
		for (int k = 0; k < suffixes.size(); k++){
			std::string kmer_to = prefix + suffixes[k];
			int state_to = kmer_to_state[kmer_to];
			transitions[state_from][state_to] = LogNum(prob_skip);
			inverse_transitions[state_to][state_from] = LogNum(prob_skip);
		}
	}
	//normal transitions
	double normal_trans_prob = (1 - prob_skip - prob_stay) / 4;
	for (std::map<std::string, int>::iterator it = kmer_to_state.begin(); it != kmer_to_state.end(); ++it){
		std::string kmer_from = it -> first;
		int state_from = it -> second;
		std::string prefix = kmer_from.substr(1,kmer_size - 1);
		for (std::set<char>::iterator it = bases.begin(); it != bases.end(); ++it){
			std::string kmer_to = prefix + (*it);
			int state_to = kmer_to_state[kmer_to];
			transitions[state_from][state_to] = LogNum(normal_trans_prob);
			inverse_transitions[state_to][state_from] = LogNum(normal_trans_prob);
		}
	}
}

std::vector<int> HMM::compute_viterbi_path(std::vector<double> event_sequence){
	Matrix<LogNum>viterbi_matrix;
	Matrix<int> back_ptr;
	for (int i = 0; i < states.size(); i++){
		std::vector<LogNum>row(event_sequence.size(), 0);
		viterbi_matrix.push_back(row);
		std::vector<int>row2(event_sequence.size(), 0);
		back_ptr.push_back(row2);
	}
	viterbi_matrix[0][0] = 1;

	for (int i = 1; i < event_sequence.size(); i++){
		for (int l = 0; l < states.size(); l++){
			LogNum m(0.0);
			for (int k = 0; k < states.size(); k++){
				if (viterbi_matrix[k][i - 1] + transitions[k][l] > m){
					m = viterbi_matrix[k][i - 1] * transitions[k][l];
					back_ptr[i][l] = k;
				}
			}
			viterbi_matrix[l][i] = m + states[l].get_emission_probability(event_sequence[i]);
		}
	}
	LogNum m(0.0);
	int last_state = 0;
	for (int k = 0; k < states.size(); k++){
		if (m < viterbi_matrix[k][event_sequence.size() - 1]){
			m = viterbi_matrix[k][event_sequence.size() - 1];
			last_state = k;
		}
	}
	std::vector<int>state_sequence;
	state_sequence.push_back(last_state);
	int prev_state = last_state;
	for (int i = event_sequence.size() - 1; i >= 1; i--){
		int s = back_ptr[i][prev_state];
		state_sequence.push_back(s);
		prev_state = s;
	}
	reverse(state_sequence.begin(), state_sequence.end());
	return state_sequence;

}

Matrix<LogNum> HMM::compute_forward_matrix(std::vector<double> event_sequence){
	Matrix<LogNum>fwd_matrix;
	for (int i = 0; i < states.size(); i++){
		std::vector<LogNum>row(event_sequence.size(), 0);
		fwd_matrix.push_back(row);
	}

	fwd_matrix[0][0] = 1;

	for (int i = 1; i < event_sequence.size(); i++){
		for (int l = 0; l < states.size(); l++){
			LogNum sum(0.0);
			for (int k = 0; k < states.size(); k++){
				sum += fwd_matrix[k][i - 1] + transitions[k][l];
			}
			fwd_matrix[l][i] = sum + states[l].get_emission_probability(event_sequence[i]);
		}
	}
	return fwd_matrix;
}

std::vector<char> HMM::translate_to_bases(std::vector<int>state_sequence){
	int prev_state = state_sequence[0];
	std::vector<char> dna_seq;
	for (int i = 1; i < state_sequence.size(); i++){
		int curr_state = state_sequence[i];
		if (prev_state == curr_state) continue;
		for (int prefix = 1; prefix <= kmer_size; prefix++){
			std::string prev_suffix = (states[prev_state].kmer_label).substr(prefix, kmer_size - prefix);
			std::string curr_prefix = (states[curr_state].kmer_label).substr(0, kmer_size - prefix);
			if (prev_suffix == curr_prefix){
				std::string appendage = (states[curr_state].kmer_label).substr(kmer_size - prefix, prefix);
				for (int k = 0; k < appendage.size(); k++){
					dna_seq.push_back(appendage[k]);
				}
			}
		}
	}
	return dna_seq;

}

Matrix<int> HMM::generate_samples(int num_of_samples, Matrix<LogNum>&forward_matrix){
	Matrix<int>res;
	return res;
}