#!/bin/bash


## Constant search
# for ad_c in $(seq 0.05 0.05 0.4)
# do
#  for im_c in $(seq 0.05 0.05 0.4)
#  do
#      python -u zero_shot_sim.py --gpu_id 1 --seed 10 --img_decay 0.9 --aud_decay 0.97 --img_const $im_c --aud_const $ad_c \
#       --lr 0.001 --tp 1.0 --batch 500 --feature_batch 8 --pd 128 --epochs 100 2>&1 | tee -a GELU-zero-shot-sim-search-pd128-best.log
#  done
# done



# ### best acc for pd=128 1 to 7: 67.4%   8 and 9 acc is 88.5%. 
# python -u zero_shot_sim.py --gpu_id 1 --seed 10 --img_decay 0.9 --aud_decay 0.97 --img_const 0.3 --aud_const 0.1 \
#       --lr 0.001 --tp 1.0 --batch 500 --feature_batch 8 --pd 64 --epochs 100 2>&1 | tee -a GELU-zero-shot-sim-search-pd64-best.log


### best acc for pd=64 1 to 7: 66.14%   8 and 9 acc is 88%. 
python -u zero_shot_sim.py --gpu_id 1 --seed 10 --img_decay 0.9 --aud_decay 0.97 --img_const 0.3 --aud_const 0.1 \
      --lr 0.001 --tp 1.0 --batch 500 --feature_batch 8 --pd 64 --epochs 100 2>&1 | tee -a GELU-zero-shot-sim-search-pd64-best.log
