#!/bin/bash

python -u zero_shot_train.py --gpu_id 1 --lr 0.001 --tp 1.0 --batch 8  --pd 64 --epochs 100 2>&1 | tee -a trainable_lsm_zero_shot.log
