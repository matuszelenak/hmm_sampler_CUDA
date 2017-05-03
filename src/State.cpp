
#include "State.h"

void State::setParams(double m, double s, double sil){
	stdv = s;
	mean = m;
	corrected_mean = m;
	corrected_stdv = s;
	silent = sil;
}

bool State::isSilent(){
	return silent;
}

LogNum State::get_emission_probability(double emission){
	if (silent) return LogNum(0.0);
	double frac = (emission - corrected_mean) / corrected_stdv;
	return LogNum((1 / (corrected_stdv * sqrt(2 * M_PI))) * exp(-0.5 * frac * frac));
}