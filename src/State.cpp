#include <vector>
#include <string>
#include <cmath>

class State{
private:
	std::string kmer_label;
	double mean;
	double stdv;
	bool silent;
public:
	bool isSilent();
	void setLabel(std::string l);

void setLabel(std::string l){
	kmer_label = l;
}

void setParams(double m, double s, double sil){
	stdv = s;
	mean = m;
	silent = sil;
}

bool isSilent(){
	return silent;
}

LogNum get_emission_probability(double emission){
	double frac = (emission - mean) / stdv;
	return Log2Num((1 / (stdv * sqrt(2 * M_PI))) * exp(-0.5 * frac * frac));
}

}