#!/bin/bash

# ## vth
# for ad_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
# do
#   for im_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
#   do
#     python -u zero_shot_train.py --gpu_id 0 --img_vth $im_v --aud_vth $ad_v --lr 0.001 --tp 1.0 --batch 8 --pd 64 --epochs 50 2>&1 | tee -a trainable_vth.log
#   done
# done


# ## decay
# for ad_d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
# do
#   for im_d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
#   do
#       python -u zero_shot_train.py --gpu_id 0 --img_decay $im_d --aud_decay $ad_d --lr 0.001 --tp 1.0 --batch 8 --pd 64 \
#       --epochs 50 2>&1 | tee -a trainable_decay.log
#   done
# done




# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0
# do
# for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0
# do
#     python -u zero_shot_prototype_train.py --gpu_id 0 --lr 0.001 --temperature 0.01 --batch 512 \
#         --pd 256 --epochs 100 --eeg_const $c1 --vis_const $c2 2>&1 | tee -a log_files/trainable_lsm_protype_constant.log
# done
# done



# for t in 0.0001 0.001 0.01 0.1 1.0 10.0
# do
# # EEG_clip/EEG_to_Image/zero_shot_prototype_train.py
#     python -u zero_shot_prototype_train.py --gpu_id 0 --lr 0.001 --temperature $t --batch 512 \
#         --pd 256 --epochs 100 2>&1 | tee -a log_files/trainable_lsm_protype_temperature.log
# done

# ## vth
# for ee_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
# do
#   for im_v in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
#   do
#     python -u zero_shot_prototype_train.py --gpu_id 0 --vis_vth $im_v --eeg_vth $ee_v \
#     --lr 0.001 --epochs 100 2>&1 | tee -a log_files/trainable_lsm_protype_vth.log
#   done
# done


python -u zero_shot_prototype_train.py --gpu_id 0 --vis_vth 1.0 --eeg_vth 6.0 \
    --lr 0.001 --epochs 100 2>&1 | tee -a log_files/trainable_protype_best.log

