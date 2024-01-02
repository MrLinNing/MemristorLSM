#!/bin/bash




# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0
# do
#         for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0
#         do
#         python -u zero_shot_train.py --gpu_id 0 --lr 0.001 --temperature 0.01 --batch 512 \
#                 --pd 256 --epochs 100 --eeg_const $c1 --vis_const $c2 2>&1 | tee -a log_files/trainable_lsm_clip_256pd.log
#         done
# done


python -u zero_shot_train.py --gpu_id 0 --lr 0.001 --temperature 0.01 --batch 512 \
                --pd 256 --epochs 100 --eeg_const 10 --vis_const 0.01 2>&1 | tee -a log_files/trainable_lsm_clip_256pd_best.log

# ## vth
# for ad_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
# do
#   for im_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
#   do
#     python -u zero_shot_train.py --gpu_id 0 --eeg_vth $im_v --vis_vth $ad_v \
#     --lr 0.001 --tp 1.0 --batch 8 --pd 256 --epochs 50 2>&1 | tee -a trainable_vth_256pd.log
#   done
# done


# ## decay
# for ad_d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
# do
#   for im_d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
#   do
#       python -u zero_shot_train.py --gpu_id 0 --eeg_decay $im_d --vis_decay $ad_d --lr 0.001 --tp 1.0 --batch 8 --pd 256 \
#       --epochs 50 2>&1 | tee -a trainable_decay.log
#   done
# done

# for t in 0.0001 0.001 0.01 0.1 1.0 10.0
# do
#     python -u zero_shot_train.py --gpu_id 0 --lr 0.001 --temperature $t --batch 512 \
#         --pd 64 --epochs 100 2>&1 | tee -a log_files/trainable_lsm_clip_256pd.log
# done


# 
# Finally In Top1-acc for E-MNIST search EEG: 17.272727272727273
# Finally In Top1-acc for EEG search E-MNIST: 21.515151515151516
# Finally Out Top1-acc for E-MNIST search EEG: 33.33333333333333
# Finally Out Top1-acc for EEG search E-MNIST: 61.66666666666667
# python -u zero_shot_train.py --gpu_id 0 --lr 0.001 --temperature 0.01 --batch 512 \
#         --pd 64 --epochs 100 --eeg_const 0.5 --vis_const 0.1 2>&1 | tee -a log_files/trainable_lsm_clip_best.log


# for bt in 16 32 64 128 256
# do
# python -u zero_shot_train.py --gpu_id 0 --lr 0.001 --temperature 0.01 --batch $bt \
#         --pd 64 --epochs 100 --eeg_const 0.5 --vis_const 0.1 2>&1 | tee -a log_files/trainable_lsm_clip_batch.log

# done