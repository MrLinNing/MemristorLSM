#!/bin/bash


### Finally 67.41%
python -u train_srnn_snn.py --gpu_id 1 --ts 70 --Tw 70 --const 1.0 --batch 128 --Decay_lsm 0.9 --Decay 0.99 --Vth 0.3 --lr 0.0002 2>&1 | tee -a train_srnn_snn.log