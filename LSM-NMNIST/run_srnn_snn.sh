#!/bin/bash

### 93.92 %
python -u train_srnn_snn.py --gpu_id 1 --const 0.03 --Vth 0.3 2>&1 | tee -a train_srnn_snn.log