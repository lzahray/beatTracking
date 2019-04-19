#!/bin/bash
set -e
set -x
SOFTMAX_FOLDER=$1
TO_SAVE_FOLDER=$2

for filename in $SOFTMAX_FOLDER/*; do
    echo "$filename"
    ../HMM/HMM_ViterbiEstimation-2 ../HMM/param_HMM.txt 0.95 0.01 $filename $TO_SAVE_FOLDER/${filename#$SOFTMAX_FOLDER/}
done
