#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for i in "${arr[@]}";
do
    echo "Running CIFAR. Anomaly Class: $i "
    python train.py --dataset cifar10 --batchsize 128 --isize 32 --niter 25 --abnormal_class $i --model skipganomaly --gen SwinUnet
done
exit 0
