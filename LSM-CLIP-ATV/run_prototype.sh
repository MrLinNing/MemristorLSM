#!/bin/bash


# #####  Constant search
# for ad_c in $(seq 0.1 0.05 0.9)
# do
#  for im_c in $(seq 0.1 0.05 0.9)
#  do
#     python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
#     --img_const $im_c --aud_const $ad_c \
#     --pd 64 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-search.log
#  done
# done



# ## Best ACC:  8 and 9: 81.7%   1~7 : 50.3%
# python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
#     --img_const 0.35 --aud_const 0.6 \
#     --pd 64 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-best.log



#####  Constant search for 1024 projection dim
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
    --img_const $im_c --aud_const $ad_c \
    --pd 1024 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-search_1024.log
 done
done

#####  Constant search for 512 projection dim
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
    --img_const $im_c --aud_const $ad_c \
    --pd 512 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-search_512.log
 done
done

#####  Constant search for 256 projection dim
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
    --img_const $im_c --aud_const $ad_c \
    --pd 256 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-search_256.log
 done
done


#####  Constant search for 128 projection dim
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype.py --img_decay 0.9 --aud_decay 0.97 \
    --img_const $im_c --aud_const $ad_c \
    --pd 128 --gpu_id 1 --seed 10 2>&1 | tee -a zero-shot-prototype-search_128.log
 done
done