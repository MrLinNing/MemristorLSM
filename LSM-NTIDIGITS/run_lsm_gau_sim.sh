#!/bin/bash


#for ct in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#  python -u lsm_sim_gaussian.py --seed 10 --ts 129 --Tw 129 --Decay_lsm 0.97 --const $ct --Vth 5.3 \
#   --lr 0.0006 --ann_batch 256 --scale 100 --ARCHI "64,200,11" 2>&1 | tee -a lsm_sim.log
#done

python -u lsm_sim_gaussian.py --seed 10 --ts 129 --Tw 129 --Decay_lsm 0.97 --const 0.08 --Vth 5.3 \
   --lr 0.0006 --ann_batch 256 --scale 100 --ARCHI "64,200,11" 2>&1 | tee -a lsm_sim.log