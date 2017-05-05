#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#include <random>
#include <chrono>

#include <cstdio>
#include <boost/log/trivial.hpp>

#include "State.h"
#include "LogNum.h"
#include "HMM.h"
#include "gpu_algorithms.h"

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

std::vector<std::string>HMM::generate_suffixes(){
	std::vector<std::string>res;
	for (std::set<char>::iterator it = bases.begin(); it != bases.end(); ++it){
		std::string acc;
		acc.push_back(*it);
		for (std::set<char>::iterator it2 = bases.begin(); it2 != bases.end(); ++it2){
			acc.push_back(*it2);
			res.push_back(acc);
			acc.pop_back();
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
		if (s[i] != delimiter) acc.push_back(s[i]);
	}
	if (acc.size() > 0) res.push_back(acc);
	return res;
}

void HMM::load_model_params(std::string filename){
	BOOST_LOG_TRIVIAL(info) << "Loading pore model parameters from " << filename << "...";
	auto start = system_clock::now();

	states.clear();
	std::ifstream model_file(filename);
	std::string line;
	std::string kmer_label;
	while(getline(model_file, line)){
		std::vector<std::string> line_elements = split_string(line, '\t');
		kmer_label = line_elements[0];
		std::size_t offset = 0;
		double mean = std::stod(&line_elements[1][0], &offset);
		double stdv = std::stod(&line_elements[2][0], &offset);
		State state;
		state.kmer_label = kmer_label;
		state.setParams(mean, stdv);
		kmer_to_state.insert(std::pair<std::string, int>(kmer_label, states.size()));
		states.push_back(state);
		for (int i = 0; i < kmer_label.size(); i++){
			bases.insert(kmer_label[i]);
		}
	}
	kmer_size = kmer_label.size();
	model_file.close();
	num_of_states = states.size();
	init_transition_prob = LogNum(1/num_of_states);
	BOOST_LOG_TRIVIAL(info) << "Loaded model with " << (int)states.size()
							<< " states and kmer size " << kmer_size 
							<< " in " << duration_cast<milliseconds>(system_clock::now() - start).count() << " ms";
}

void HMM::adjust_scaling(std::vector<double>& event_sequence){
	BOOST_LOG_TRIVIAL(info) << "Adjusting the scaling of pore model parameters...";
	double dmax = DBL_MIN;
	double dmin = DBL_MAX;
	double mmax = DBL_MIN;
	double mmin = DBL_MAX;
	double sc, sh;
	for (int i = 0; i < event_sequence.size(); i++){
		dmax = std::max(dmax, event_sequence[i]);
		dmin = std::min(dmin, event_sequence[i]);
	}
	for (int i = 1; i < states.size(); i++){
		mmax = std::max(mmax, states[i].mean);
		mmin = std::min(mmin, states[i].mean);
	}
	sc = (dmax - dmin) / (mmax - mmin);
	sh = dmax - mmax*sc;
	BOOST_LOG_TRIVIAL(info) << "Done. Scale = "<< sc << ", shift = " << sh;
	for (int i = 1; i < states.size(); i++){
		states[i].corrected_mean = states[i].corrected_mean*sc + sh;
	}
}

void HMM::set_stay_prob(double prob){
	prob_stay = prob;
}

void HMM::set_skip_prob(double prob){
	prob_skip = prob;
}

int HMM::kmer_overlap(std::string from, std::string to){
	int count = 0;
	for (int prefix = 0; prefix < from.size(); prefix++){
		std::string from_suffix = from.substr(prefix, from.size() - prefix);
		std::string to_prefix = to.substr(0, from.size() - prefix);
		if (from_suffix == to_prefix){
			return count;
		}
		count++;
	}
	return count;
}

void HMM::compute_transitions(){
	BOOST_LOG_TRIVIAL(info) << "Computing state transition probabilites...";
	auto start = system_clock::now();

	transitions.resize(states.size(), std::vector<LogNum>(states.size(), LogNum(0.0)));
	inverse_neighbors.resize(states.size());

	for (int state_from = 0; state_from < states.size(); state_from++){
		std::vector<int>normal_states;
		std::vector<int>skip_states;
		for (int state_to = 0; state_to < states.size(); state_to ++){
			std::string kmer_from = states[state_from].kmer_label;
			std::string kmer_to = states[state_to].kmer_label;
			int overlap = kmer_overlap(kmer_from, kmer_to);
			switch(overlap){
				case 0 : {
					transitions[state_from][state_to] = LogNum(prob_stay);
					inverse_neighbors[state_to].push_back({state_from, LogNum(prob_stay)});
					break;
				}
				case 1 : {
					normal_states.push_back(state_to);
					break;
				}
				case 2 : {
					skip_states.push_back(state_to);
					break;
				}
			}
		}
		for (int i = 0; i < normal_states.size(); i++){
			transitions[state_from][normal_states[i]] = LogNum((1 - prob_skip - prob_stay) / normal_states.size());
			inverse_neighbors[normal_states[i]].push_back({state_from, LogNum((1 - prob_skip - prob_stay) / normal_states.size())});
		}
		for (int i = 0; i < skip_states.size(); i++){
			transitions[state_from][skip_states[i]] = LogNum(prob_skip / skip_states.size());
			inverse_neighbors[skip_states[i]].push_back({state_from, LogNum(prob_skip / skip_states.size())});
		}
	}
	BOOST_LOG_TRIVIAL(info) << "Transition computation done in "<< duration_cast<milliseconds>(system_clock::now() - start).count() << " ms";
}

std::vector<int> HMM::compute_viterbi_path(std::vector<double>& event_sequence){

	BOOST_LOG_TRIVIAL(info) << "Running viterbi" << "\n";
	auto start = system_clock::now();

	Matrix<LogNum>viterbi_matrix(event_sequence.size(), std::vector<LogNum>(states.size(), LogNum(0.0)));
	Matrix<int> back_ptr(event_sequence.size(), std::vector<int>(states.size(), 0));
	for (int i = 0; i < states.size(); i++){
		viterbi_matrix[0][i] = init_transition_prob + states[i].get_emission_probability(event_sequence[0]);
	}
	for (int i = 1; i < event_sequence.size(); i++){
		for (int l = 0; l < states.size(); l++){
			LogNum m(0.0);
			for (int j = 0; j < inverse_neighbors[l].size(); j++){
				std::pair<int, LogNum>p = inverse_neighbors[l][j];
				int k = p.first;
				LogNum t_prob = p.second;
				if (viterbi_matrix[i - 1][k] + t_prob > m){
					m = viterbi_matrix[i - 1][k] + t_prob;
					back_ptr[i][l] = k;
				}
			}
			viterbi_matrix[i][l] = m + states[l].get_emission_probability(event_sequence[i]);
		}
	}

	LogNum m(0.0);
	int last_state = 0;
	for (int k = 0; k < states.size(); k++){
		if (m < viterbi_matrix[event_sequence.size() - 1][k]){
			m = viterbi_matrix[event_sequence.size() - 1][k];
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
	BOOST_LOG_TRIVIAL(info) << "Viterbi path complete in " << duration_cast<milliseconds>(system_clock::now() - start).count() << " ms";
	return state_sequence;
}

Matrix<std::vector<double> > HMM::compute_forward_matrix(std::vector<double>& event_sequence){
	BOOST_LOG_TRIVIAL(info) << "Computing forward matrix";
	auto start = system_clock::now();

	Matrix<LogNum>fwd_matrix(event_sequence.size(), std::vector<LogNum>(states.size(), LogNum(0.0)));

	Matrix<std::vector<double> >probability_weights(event_sequence.size(), std::vector<std::vector<double> >(states.size()));
	for (int i = 0; i < states.size(); i++){
		fwd_matrix[0][i] = init_transition_prob + states[i].get_emission_probability(event_sequence[0]);
	}
	for (int i = 1; i < event_sequence.size(); i++){
		for (int l = 0; l < states.size(); l++){
			LogNum sum(0.0);
			std::vector<LogNum>probabilites;
			for (int j = 0; j < inverse_neighbors[l].size(); j++){
				std::pair<int, LogNum>p = inverse_neighbors[l][j];
				int k = p.first;
				LogNum t_prob = p.second;
				sum += fwd_matrix[i - 1][k] + t_prob;
				probabilites.push_back(fwd_matrix[i - 1][k] + t_prob);
			}
			fwd_matrix[i][l] = sum + states[l].get_emission_probability(event_sequence[i]);
			LogNum prob_sum = std::accumulate(probabilites.begin(), probabilites.end(), LogNum(0.0));
			probability_weights[i][l].resize(probabilites.size());
			if (!prob_sum.isZero()){
				std::transform(probabilites.begin(),
								probabilites.end(),
								probability_weights[i][l].begin(),
								[prob_sum](LogNum x){
									return (x.value()/prob_sum.value());});
			}
		}
		BOOST_LOG_TRIVIAL(info) << "Row " << i << " done out of " << event_sequence.size();
	}
	BOOST_LOG_TRIVIAL(info) << "Forward matrix calculation done in " << duration_cast<milliseconds>(system_clock::now() - start).count() << " ms";
	return probability_weights;
}

std::vector<char> HMM::translate_to_bases(std::vector<int>&state_sequence) const{
	int prev_state = state_sequence[0];
	std::vector<char> dna_seq;
	for (int i = 0; i < states[prev_state].kmer_label.size(); i++) dna_seq.push_back(states[prev_state].kmer_label[i]);
	for (int i = 1; i < state_sequence.size(); i++){
		int curr_state = state_sequence[i];
		for (int prefix = 1; prefix <= kmer_size; prefix++){
			std::string prev_suffix = (states[prev_state].kmer_label).substr(prefix, kmer_size - prefix);
			std::string curr_prefix = (states[curr_state].kmer_label).substr(0, kmer_size - prefix);
			if (prev_suffix == curr_prefix){
				std::string appendage = (states[curr_state].kmer_label).substr(kmer_size - prefix, prefix);
				for (int k = 0; k < appendage.size(); k++){
					dna_seq.push_back(appendage[k]);
				}
				break;
			}
		}
		prev_state = curr_state;
	}
	return dna_seq;

}

std::vector<int> HMM::backtrack_sample(int last_state, int l, Matrix<std::vector<double> > &prob_weights, std::default_random_engine gen){
	std::vector<int>sample;
	int curr_state = last_state;
	while (l >= 0){
		sample.push_back(curr_state);
		std::discrete_distribution<>sample_next_state(prob_weights[l][curr_state].begin(), prob_weights[l][curr_state].end());
		curr_state = inverse_neighbors[curr_state][sample_next_state(gen)].first;
		l--;
	}
	std::reverse(sample.begin(), sample.end());
	return sample;
}

Matrix<int> HMM::generate_samples(int num_of_samples, std::vector<double>&event_sequence, int seed){
	BOOST_LOG_TRIVIAL(info) << "Generating "<< num_of_samples <<"samples";
	auto start = system_clock::now();
	Matrix<int>res;
	Matrix<std::vector<double> > prob_weights = compute_forward_matrix(event_sequence);
	Matrix<double> last_states = prob_weights.back();
	std::vector<double> last_state_weights(last_states.size());
	std::transform(last_states.begin(), last_states.end(), last_state_weights.begin(),
				  [this](std::vector<double>x){
				  	double s = std::accumulate(x.begin(), x.end(), 0.0);
				  	return s/num_of_states;
				  });

	std::default_random_engine gen(seed);
	std::discrete_distribution<> choose_last_state(last_state_weights.begin(), last_state_weights.end());
	
	for (int i = 0; i < num_of_samples; i++){
		res.push_back(backtrack_sample(choose_last_state(gen), (int)event_sequence.size() - 1, prob_weights, gen));
	}
	BOOST_LOG_TRIVIAL(info) << "Generating "<< num_of_samples <<"samples took " << duration_cast<milliseconds>(system_clock::now() - start).count() << " ms";
	return res;
}