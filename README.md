# Sampling from MinION Reads

## Introduction

The MinION DNA sequencing platform produces long reads with high error
rates. One of the reasons for high error rates is that electrical signals produced
by the sequencing machine need to be first translated into DNA sequences by a
process called base calling, and this process is error prone. An alternative way
of interpreting these signals is to generate multiple samples from a posterior
sequence distributions defined by a hidden Markov model (HMM) representing
the properties of the sequencing process. The goal of this project is to speed up
this sampling by employing GPUs.

## Installation

For a successful compilation you should have the following packages installed
- Boost for C++
- HDF5

Compile using this command in the _src_ folder
```
g++ *.cpp --std=c++11 -O2 -o hmm_sampler -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK -I /usr/include/hdf5/serial/ -lhdf5_serial
```
Run with
```
./hmm_sampler
```
