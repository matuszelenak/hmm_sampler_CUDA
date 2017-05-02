#ifndef LOGNUM_H
#define LOGNUM_H

#include <cmath>
#include <limits>

class LogNum{
public:
	double exponent;
	//LogNum(double n);

	bool isZero() const;

	double exponentiate();

	LogNum(double n);

	LogNum& operator+=(const LogNum& a);

	LogNum& operator*=(const LogNum& a);

	bool operator<(const LogNum& a)const;
	bool operator>(const LogNum& a)const;

	friend LogNum operator+(LogNum &a, const LogNum &b);

	friend LogNum operator*(LogNum &a, const LogNum &b);

};

#endif