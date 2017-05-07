#!/bin/bash
clear
clear
nvcc -c gpu_algorithms.cu -arch sm_35
g++ -c test_main_gpu.cpp HMM.cpp State.cpp LogNum.cpp --std=c++11  -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK
nvcc -arch sm_35 test_main_gpu.o HMM.o State.o LogNum.o gpu_algorithms.o -o test_gpu  -I /usr/include/hdf5/serial/ -lhdf5_serial -lboost_program_options -lpthread -lboost_log -DBOOST_LOG_DYN_LINK