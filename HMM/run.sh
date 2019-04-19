#! /bin/bash

#To run the Viterbi algorithm (random parameters)
#./HMM_ViterbiEstimation param_HMM.txt ex_data.txt ex_result.txt

#To run the Viterbi algorithm (trained parameters, random data)
./HMM_ViterbiEstimation-2 param_HMM.txt 0.95 0.01 ex_data.txt ex_result.txt
