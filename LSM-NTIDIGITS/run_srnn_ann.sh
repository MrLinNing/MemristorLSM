#!/bin/bash


## 82.70%
python -u train_srnn_ann.py --gpu_id 1 --ts 90 --Tw 90 --const 1.0 --Decay_lsm 0.9 --Vth 0.6 2>&1 | tee -a train_srnn_ann.log











