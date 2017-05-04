#ifndef STATE_H
#define STATE_H

#include <vector>
#include <string>
#include <cmath>

#include "LogNum.h"

class State{
public:
	std::string kmer_label;
	double mean;
	double stdv;
	double corrected_mean;
	double corrected_stdv;
	void setParams(double m, double s);
	LogNum get_emission_probability(double emission);

};

#endif