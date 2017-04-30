import math

def logsum(a,b):
	if a > b :
		return a + math.log1p(exp(b - a))
	else:
		return b + math.log1p(exp(a - b))