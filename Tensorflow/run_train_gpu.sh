#!/bin/sh
# 10 20 50 100
rm -r logs/
rm -r results/
rm -r ./*.out
mkdir logs
mkdir results

nohup python train_combined_multi_gpu.py --k 10 --start_index 0 &> 10.out &
nohup python train_combined_multi_gpu.py --k 20 --start_index 1000 &> 20.out &
nohup python train_combined_multi_gpu.py --k 50 --start_index 2000 &> 50.out &
nohup python train_combined_multi_gpu.py --k 100 --start_index 3000 &> 100.out &