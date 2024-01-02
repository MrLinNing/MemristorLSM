#!/bin/bash

### constant search
#for ct in 0.001 0.003 0.005 0.007 0.009 0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5 0.7 0.9 1.0 3.0 5.0 7.0 9.0
#do
#python -u lsm_sim_gaussian.py --gpu_id 1 --seed 10 --const $ct  --Vth 5.0 2>&1 | tee -a search_lsm_sim_guassian.log
#done

## constant search
#for ct in 0.06 0.065 0.07 0.075 0.08 0.085 0.09
#do
#python -u lsm_sim_gaussian.py --gpu_id 1 --seed 10 --const $ct  --Vth 5.0 2>&1 | tee -a search_lsm_sim_guassian.log
#done

## Finally Accuracy 89.11%
python -u lsm_sim_gaussian.py --gpu_id 1 --seed 10 --const 0.07  --Vth 5.0 2>&1 | tee -a search_lsm_sim_guassian.log
