#!/bin/bash


# #####  Constant search
# for ad_c in $(seq 0.1 0.05 0.9)
# do
#  for im_c in $(seq 0.1 0.05 0.9)
#  do
#     python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
#     --img_const $im_c --aud_const $ad_c \
#     --pd 64 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_search.log
#  done
# done

# ### Best Acc: 
# python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
#     --img_const 0.35 --aud_const 0.9 \
#     --pd 64 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_best_acc.log



#####  Constant search for 256 hidden size
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
    --img_const $im_c --aud_const $ad_c \
    --pd 1024 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_search_1024.log
 done
done

#####  Constant search for 256 hidden size
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
    --img_const $im_c --aud_const $ad_c \
    --pd 512 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_search_512.log
 done
done



#####  Constant search for 256 hidden size
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
    --img_const $im_c --aud_const $ad_c \
    --pd 256 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_search_256.log
 done
done

#####  Constant search 128 hidden size
for ad_c in $(seq 0.1 0.05 0.9)
do
 for im_c in $(seq 0.1 0.05 0.9)
 do
    python -u zero_shot_prototype_trainable.py --gpu_id 1 --lr 0.001 --tp 1.0 \
    --img_const $im_c --aud_const $ad_c \
    --pd 128 --epochs 100 2>&1 | tee -a trainable_prototype_zero_shot_search_128.log
 done
done