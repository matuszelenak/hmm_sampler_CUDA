#ifndef STATE_H
#define STATE_H

#include <vector>
#include <string>
#include <cmath>

#include "LogNum.h"

class State{
private:
	double mean;
	double stdv;
	bool silent;
public:
	std::string kmer_label;
	bool isSilent();
	void setParams(double m, double s, double sil);
	LogNum get_emission_probability(double emission);

};

#endif