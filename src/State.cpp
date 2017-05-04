
#include "State.h"

void State::setParams(double m, double s){
	stdv = s;
	mean = m;
	corrected_mean = m;
	corrected_stdv = s;
}

LogNum State::get_emission_probability(double emission){
	double frac = (emission - corrected_mean) / corrected_stdv;
	return LogNum((1 / (corrected_stdv * sqrt(2 * M_PI))) * exp(-0.5 * frac * frac));
}