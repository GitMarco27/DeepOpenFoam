#!/bin/sh
nohup python train_single_model.py --k 40 --start_index 10003 --data_path dataset_augmented --lr 0.0001 --batch_size 256 --decoder dense --arch_bool True --beta 1 &> new.out &
