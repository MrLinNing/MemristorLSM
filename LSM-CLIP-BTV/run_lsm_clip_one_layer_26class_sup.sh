#!/bin/bash



# for t in 0.0001 0.001 0.01 0.1 1.0 10.0 
# do 

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 512 \
#         --batch 22 --feature_batch 32 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb32_all_pd512_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 512 \
#         --batch 22 --feature_batch 64 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb64_all_pd512_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 512 \
#         --batch 22 --feature_batch 128 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd512_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 512 \
#         --batch 22 --feature_batch 256 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb256_all_pd512_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 512 \
#         --batch 22 --feature_batch 512 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb512_all_pd512_22_supervised.log
# done


# for t in 0.0001 0.001 0.01 0.1 1.0 10.0 
# do 

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 32 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb32_all_pd256_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 64 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb64_all_pd256_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 128 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 256 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb256_all_pd256_22_supervised.log

#     python -u train_zero_shot_clip_enlarge_suploss.py --gpu_id 1 --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 512 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb512_all_pd256_22_supervised.log
# done


# ### EEG Search E-MNIST
# ## out acc: 58.12
# ## in acc: 48.93
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised.log


# ### hidden size search

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 4096, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 4096, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_hidden.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_hidden.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 1024, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 1024, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_hidden.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 512, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 512, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_hidden.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 256, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 256, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_hidden.log


# ### data enlarge search
# for el in 4 6 8 10 12 14 15 18 20
# do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 --enlarge_eeg $el \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_el.log
# done

# ### data agumentation search
# for sd in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 
# do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#             --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#             --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#             --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std $sd \
#             --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_std.log
# done

# ### constant search for OneFC Readout
# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
# do
#     for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
#     do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const $c1 --emnist_const $c2 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_constant_onelayer.log
#     done
# done


# ### constant search for TwoFC Readout
# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
# do
#     for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
#     do
#     python -u train_zero_shot_clip_enlarge_all.py --data_num "data_26" \
#             --eeg_const $c1 --emnist_const $c2 --models 'TwoFC' --epochs 100 \
#             --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.0001 --pd 256 \
#             --batch 22 --feature_batch 128 --gpu_id 0 \
#             --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_constant_twolayer.log
#     done
# done


# ### constant search for ThreeFC Readout
# for c1 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
# do
#     for c2 in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
#     do
#     python -u train_zero_shot_clip_enlarge_all.py --data_num "data_26" \
#             --eeg_const $c1 --emnist_const $c2 --models 'ThreeFC' --epochs 100 \
#             --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.0001 --pd 256 \
#             --batch 22 --feature_batch 128 --gpu_id 0 \
#             --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_supervised_constant_threelayer.log
#     done
# done


# ### data enlarge search
# for train_el in 4 6 8 10 12 14 15 18 20
# do
# for test_el in 2 3 4 5 6 7 8 9 10
# do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --enlarge_eeg_train $train_el \
#         --enlarge_eeg_test $test_el \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_onefc_supervised_train_test_el.log
# done
# done


# ### data enlarge search
# for train_el in 4 6 8 10 12 14 15 18 20
# do
# for test_el in 2 3 4 5 6 7 8 9 10
# do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'TwoFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --enlarge_eeg_train $train_el \
#         --enlarge_eeg_test $test_el \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_twofc_supervised_train_test_el.log
# done
# done


# ### data enlarge search
# for train_el in 4 6 8 10 12 14 15 18 20
# do
# for test_el in 2 3 4 5 6 7 8 9 10
# do
#     python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'ThreeFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0.0 \
#         --enlarge_eeg_train $train_el \
#         --enlarge_eeg_test $test_el \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_fb12_pd256_22_threefc_supervised_train_test_el.log
# done
# done



# ### EEG Search E-MNIST
# ## out acc: 58.12
# ## in acc: 48.93



# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "0,1,2,3" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "1,2,3,4" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "2,3,4,5" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "3,4,5,6" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "4,5,6,7" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "5,6,7,8" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "6,7,8,9" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "7,8,9,10" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "8,9,10,11" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,13,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "9,10,11,12" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "10,11,12,13" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "11,12,13,14" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,24,25" \
#         --out_class "12,13,14,15" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24,25" \
#         --out_class "13,14,15,16" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20,21,22,23,24,25" \
#         --out_class "14,15,16,17" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20,21,22,23,24,25" \
#         --out_class "15,16,17,18" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25" \
#         --out_class "16,17,18,19" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25" \
#         --out_class "17,18,19,20" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,23,24,25" \
#         --out_class "19,20,21,22" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25" \
#         --out_class "20,21,22,23" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25" \
#         --out_class "21,22,23,24" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21" \
#         --out_class "22,23,24,25" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log

### EEG search E-MNIST 
## In-acc: 49.84
## Out-acc: 60%

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer.log


# for t in 0.0001 0.0005 0.001 0.005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0
# do
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature $t --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_temperature.log
# done


# for bc in 32 64 128 256 512 1024
# do
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch $bc --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_featurebatch.log
# done


# for pdd in 8 16 32 64 128 256 512 1024
# do
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd $pdd \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_pd.log
# done


# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 256, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 256, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 512, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 512, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 1024, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 1024, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_archi.log


# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 4096, 26" --act 'gelu' --temperature 0.001 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 4096, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_archi.log


### Best Acc
## In-Class: 50.15%
## Out-Class: 68.33%
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best.log




# for bc in 32 64 128 256 512 1024
# do
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch $bc --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_batch.log
# done


# for pdd in 8 16 32 64 128 256 512 1024
# do
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd $pdd \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_pd.log
# done


# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 256, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 256, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 512, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 512, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 1024, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 1024, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_archi.log


# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_archi.log

# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 4096, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 128 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 4096, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_archi.log


# ### Best Acc:
# ## In-Class: 52.27%
# ## Out-Class: 70.83%
# python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
#         --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
#         --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd 256 \
#         --batch 22 --feature_batch 1024 --gpu_id 0 --eeg_std 0 \
#         --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
#         --out_class "18,19,20,21" \
#         --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_batch.log


for s in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
do
python -u train_zero_shot_clip_enlarge_suploss.py --data_num "data_26" \
        --eeg_const 0.0005 --emnist_const 0.0005 --models 'OneFC' --epochs 100 \
        --emnist_ARCHI "784, 2048, 26" --act 'gelu' --temperature 0.5 --pd 256 \
        --batch 22 --feature_batch 1024 --gpu_id 0 --eeg_std $s \
        --in_class "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25" \
        --out_class "18,19,20,21" \
        --eeg_ARCHI "192, 2048, 26" 2>&1 | tee -a log_files/lsm_zero_shot_fb128_all_pd256_22_supervised_data_transfer_best_batch_std.log
done