#include <cmath>
#include <limits.h>

class LogNum{
public:
	double exponent;
	LogNum(double n);

LogNum::LogNum(double n){
	exponent = log(n);
}

bool isZero(){
	return (exponent == -HUGE_VALUE);
}

double exponentiate(){
	return exp(exponent);
}

LogNum& LogNum::operator+=(const LogNum& a){
	if (this->isZero()){
		*this = a;
	}
	else{
		if (this -> exponent > a.exponent){
			this -> exponent += log1p(exp(a.exponent - this -> exponent));
		}
		else{
			this -> exponent += log1p(exp(this -> exponent - a.exponent));
		}
	}
}
}