#!/bin/bash
clear
clear
nvcc -dc gpu_sampling.cu gpu_samplingv2.cu gpu_viterbi.cu gpu_utils.cu -arch sm_35
g++ -c HMM.cpp State.cpp LogNum.cpp main.cpp --std=c++11 -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK
nvcc -arch sm_35 -o  hmm_sampler HMM.o State.o LogNum.o main.o gpu_sampling.o gpu_samplingv2.o gpu_viterbi.o gpu_utils.o -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK