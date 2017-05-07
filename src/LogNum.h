#ifndef LOGNUM_H
#define LOGNUM_H

#include <cmath>
#include <cfloat>

const double kEpsilon = 1.0e-15;

class LogNum{
public:
	double exponent;

	bool isZero() const;

	double value() const;

	LogNum();

	LogNum(double n);

	LogNum& operator+=(const LogNum& a);

	LogNum& operator*=(const LogNum& a);

	LogNum& operator/=(const LogNum& a);

	bool operator<(const LogNum& a)const;
	bool operator>(const LogNum& a)const;

	friend LogNum operator+(LogNum &a, const LogNum &b);

	friend LogNum operator*(LogNum &a, const LogNum &b);

	friend LogNum operator/(LogNum &a, const LogNum &b);

	bool operator==(const LogNum& num) const;

	double get_exponent() const;

};

#endif