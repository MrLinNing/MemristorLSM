#!/bin/bash

###  75.54%
python -u train_rnn.py --gpu_id 1 --Tw 30 2>&1 | tee -a train_RNN.log


#for t in 10 15 20 25 30 35 40 45 50
#do
#  python -u train_rnn.py --gpu_id 0 --Tw $t 2>&1 | tee -a train_RNN.log
#done