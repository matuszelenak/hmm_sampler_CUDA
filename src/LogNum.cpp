#include "LogNum.h"

LogNum::LogNum(double n){
	exponent = log(n);
}

bool LogNum::isZero() const{
	return (exponent == -HUGE_VAL);
}

double LogNum::value() const{
	if (isZero()) return 0;
	return exp(exponent);
}

LogNum operator+(LogNum &a, const LogNum &b){
	LogNum r(0.0);
	r.exponent = a.exponent;
	r+=b;
	return r;
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
	return *this;
}

LogNum operator*(LogNum &a, const LogNum &b){
	LogNum r(0.0);
	r.exponent = a.exponent;
	r*=b;
	return r;
}

LogNum operator/(LogNum &a, const LogNum &b){
	LogNum r(0.0);
	r.exponent = a.exponent;
	r/=b;
	return r;
}

LogNum& LogNum::operator*=(const LogNum& a){
	if (this->isZero() || a.isZero()){
		this -> exponent = - HUGE_VAL;
	}
	else{
		this -> exponent += a.exponent;
	}
	return *this;
}

LogNum& LogNum::operator/=(const LogNum& a){
	if (!this -> isZero()) this -> exponent -= a.exponent;
	return *this;
}

bool LogNum::operator<(const LogNum& a)const{
	if (isZero() && !a.isZero()) return true;
	if (a.isZero()) return false;
	return exponent < a.exponent;
}

bool LogNum::operator>(const LogNum& a)const{
	if (!isZero() && a.isZero()) return true;
	if (a.isZero()) return true;
	return exponent > a.exponent;
}

bool LogNum::operator==(const LogNum& num) const {
  double this_val = this->value();
  double num_val = num.value();
  return fabs(this_val - num_val) < kEpsilon;
}

double LogNum::get_exponent() const{
	return exponent;
}