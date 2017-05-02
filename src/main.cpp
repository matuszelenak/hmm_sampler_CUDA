#include <iostream>
#include <fstream>

#include "HMM.h"

int main(int argc, char const *argv[])
{
	HMM hmm;
	hmm.loadModelParams("r73.c.p1.006.ont.model");
	return 0;
}