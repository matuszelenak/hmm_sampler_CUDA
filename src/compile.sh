#!/bin/bash
clear
clear
nvcc -c gpu_algorithms.cu
g++ -c HMM.cpp State.cpp LogNum.cpp main.cpp --std=c++11 -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK
nvcc HMM.o State.o LogNum.o main.o gpu_algorithms.o -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK