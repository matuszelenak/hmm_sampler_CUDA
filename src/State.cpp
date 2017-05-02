
#include "State.h"

void State::setParams(double m, double s, double sil){
	stdv = s;
	mean = m;
	silent = sil;
}

bool State::isSilent(){
	return silent;
}

LogNum State::get_emission_probability(double emission){
	double frac = (emission - mean) / stdv;
	return LogNum((1 / (stdv * sqrt(2 * M_PI))) * exp(-0.5 * frac * frac));
}