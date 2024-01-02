#!/bin/bash

### best acc is 96.23%
python -u train_rnn.py --gpu_id 1 --epochs 100 --Tw 50 2>&1 | tee -a train_rnn.log