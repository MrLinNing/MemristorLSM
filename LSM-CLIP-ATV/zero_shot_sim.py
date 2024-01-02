import torch
import time
from datetime import date, datetime
import os
import argparse
from utils import MyCenterCropDataset
from utils import AvgMeter
from torch.autograd import Variable
from utils import CustomTensorDataset
from ebdataset.audio import NTidigits
import random
import numpy as np
from quantities import ms, second
from models import Contrastive_Simple, NM_simulation_Gau_LSM_Model, NT_simulation_Gau_LSM_Model
from models import Contrastive_twolayers, Contrastive_threelayers
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True





def parse_args():
    parser = argparse.ArgumentParser()
    ### Hyper-parameters
    parser.add_argument('--batch', type=int, default=1000,
                        help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate ')

    ### Parameters for LSM
    parser.add_argument('--pd', type=int, default=256,
                        help='projection dimention')
    parser.add_argument('--tp', type=float, default=1.0,
                        help='temperature factor for cosine similarities')

    parser.add_argument('--img_const', type=float, default=0.5,
                        help='image encoder weight constant')
    parser.add_argument('--aud_const', type=float, default=0.5,
                        help='audio encoder weight constant')

    parser.add_argument('--img_decay', type=float, default=0.9,
                        help='image encoder decay')
    parser.add_argument('--aud_decay', type=float, default=0.97,
                        help='audio encoder decay')

    parser.add_argument('--img_vth', type=float, default=5.0,
                        help='image vth')
    parser.add_argument('--aud_vth', type=float, default=5.3,
                        help='image vth')

    parser.add_argument('--feature_batch', type=int, default=16,
                        help='encoding feature batch size')
    parser.add_argument('--ts', type=int, default=129,
                        help='time_step')
    parser.add_argument('--Dt', type=int, default=1,
                        help='Integration window')
    
    ### Parameters for Readout
    parser.add_argument('--models', type=str, default='OneFC',
                        help='OneFC, TwoFC, ThreeFC')
    
    ### Parameters for activation in Readout layer
    parser.add_argument('--act', type=str, default='gelu',
                        help='relu, sigmoid, tanh, gelu')

    ## configurations
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

SAVE_DIR = 'zero-shot-simulation-checkpoint'


####set seed###
setup_seed(options.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR,
f'ACT{options.act}_FC{options.models}_seed{options.seed}_audvth{options.aud_vth}_imgvth{options.img_vth}_audconst{options.aud_const}_imgconst{options.img_const}_auddecay{options.aud_decay}_imgdecay{options.img_decay}_temperature{options.tp}_projectiondimention{options.pd}_featurebatch{options.feature_batch}_batch{options.batch}_lr{options.lr}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')
device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")

####################### NMNIST Datasets ########################
train_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_train_data.mat'
test_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_test_data.mat'
train_dataset_crop = MyCenterCropDataset(
    train_path, 'nmnist_h', new_width=16, new_height=16)
# test data size: 10k
test_dataset_crop = MyCenterCropDataset(
    test_path, 'nmnist_r', new_width=16, new_height=16)


print('=============NMNIST train data collection=============')
nm_tr_data_0 = []
nm_tr_data_1 = []
nm_tr_data_2 = []
nm_tr_data_3 = []
nm_tr_data_4 = []
nm_tr_data_5 = []
nm_tr_data_6 = []
nm_tr_data_7 = []
nm_tr_data_8 = []
nm_tr_data_9 = []

nm_tr_label_0 = []
nm_tr_label_1 = []
nm_tr_label_2 = []
nm_tr_label_3 = []
nm_tr_label_4 = []
nm_tr_label_5 = []
nm_tr_label_6 = []
nm_tr_label_7 = []
nm_tr_label_8 = []
nm_tr_label_9 = []

for i in range(len(train_dataset_crop)):
    data = train_dataset_crop[i][0]
    # print('Image is',train_dataset_crop[i][0])
    # print('Image shape is', train_dataset_crop[i][0].shape)
    # print('label is', train_dataset_crop[i][1])
    labels = torch.argmax(train_dataset_crop[i][1],dim=0)
    # print('label is', labels.data)

    if labels == 1:
        nm_tr_data_1.append(data)
        nm_tr_label_1.append(labels)
    elif labels == 2:
        nm_tr_data_2.append(data)
        nm_tr_label_2.append(labels)
    elif labels == 3:
        nm_tr_data_3.append(data)
        nm_tr_label_3.append(labels)
    elif labels == 4:
        nm_tr_data_4.append(data)
        nm_tr_label_4.append(labels)
    elif labels == 5:
        nm_tr_data_5.append(data)
        nm_tr_label_5.append(labels)
    elif labels == 6:
        nm_tr_data_6.append(data)
        nm_tr_label_6.append(labels)
    elif labels == 7:
        nm_tr_data_7.append(data)
        nm_tr_label_7.append(labels)
    elif labels == 8:
        nm_tr_data_8.append(data)
        nm_tr_label_8.append(labels)
    elif labels == 9:
        nm_tr_data_9.append(data)
        nm_tr_label_9.append(labels)


print("nm data 1 lenth is", len(nm_tr_data_1))
print("nm data 2 lenth is", len(nm_tr_data_2))
print("nm data 3 lenth is", len(nm_tr_data_3))
print("nm data 4 lenth is", len(nm_tr_data_4))
print("nm data 5 lenth is", len(nm_tr_data_5))
print("nm data 6 lenth is", len(nm_tr_data_6))
print("nm data 7 lenth is", len(nm_tr_data_7))
print("nm data 8 lenth is", len(nm_tr_data_8))
print("nm data 9 lenth is", len(nm_tr_data_9))

print('=============NMNIST test data collection=============')
nm_te_data_0 = []
nm_te_data_1 = []
nm_te_data_2 = []
nm_te_data_3 = []
nm_te_data_4 = []
nm_te_data_5 = []
nm_te_data_6 = []
nm_te_data_7 = []
nm_te_data_8 = []
nm_te_data_9 = []

nm_te_label_0 = []
nm_te_label_1 = []
nm_te_label_2 = []
nm_te_label_3 = []
nm_te_label_4 = []
nm_te_label_5 = []
nm_te_label_6 = []
nm_te_label_7 = []
nm_te_label_8 = []
nm_te_label_9 = []

for i in range(len(test_dataset_crop)):
    data = test_dataset_crop[i][0]
    # print('Image is',train_dataset_crop[i][0])
    # print('Image shape is', train_dataset_crop[i][0].shape)
    # print('label is', train_dataset_crop[i][1])
    labels = torch.argmax(test_dataset_crop[i][1],dim=0)
    # print('label is', labels.data)

    if labels == 1:
        nm_te_data_1.append(data)
        nm_te_label_1.append(labels)
    elif labels == 2:
        nm_te_data_2.append(data)
        nm_te_label_2.append(labels)
    elif labels == 3:
        nm_te_data_3.append(data)
        nm_te_label_3.append(labels)
    elif labels == 4:
        nm_te_data_4.append(data)
        nm_te_label_4.append(labels)
    elif labels == 5:
        nm_te_data_5.append(data)
        nm_te_label_5.append(labels)
    elif labels == 6:
        nm_te_data_6.append(data)
        nm_te_label_6.append(labels)
    elif labels == 7:
        nm_te_data_7.append(data)
        nm_te_label_7.append(labels)
    elif labels == 8:
        nm_te_data_8.append(data)
        nm_te_label_8.append(labels)
    elif labels == 9:
        nm_te_data_9.append(data)
        nm_te_label_9.append(labels)

print("nm test data 1 lenth is", len(nm_te_data_1))
print("nm test data 2 lenth is", len(nm_te_data_2))
print("nm test data 3 lenth is", len(nm_te_data_3))
print("nm test data 4 lenth is", len(nm_te_data_4))
print("nm test data 5 lenth is", len(nm_te_data_5))
print("nm test data 6 lenth is", len(nm_te_data_6))
print("nm test data 7 lenth is", len(nm_te_data_7))
print("nm test data 8 lenth is", len(nm_te_data_8))
print("nm test data 9 lenth is", len(nm_te_data_9))

#########################################################  N-Tidigits Datasets ########################

dt = int(options.Dt) * ms
n_features = 64

def rec_array_to_spike_train(sparse_spike_train):
    ts = sparse_spike_train.ts * second
    ts = (ts.rescale(dt.units) / dt).magnitude
    duration = np.ceil(np.max(ts)) + 1
    spike_train = torch.zeros((n_features, duration.astype(int)))
    spike_train[sparse_spike_train.addr, ts.astype(int)] = 1

    #convert to fixed time_step
    spike_train = spike_train.unsqueeze(1)
    spike_train = spike_train.chunk(options.ts, dim = 2)

    spike_train_fixed = []
    for i in spike_train:
        spike_train_fixed.append(i.sum(2))
    spike_train_fixed = torch.cat(spike_train_fixed, dim=1)
    spike_train_fixed[spike_train_fixed > 0] = 1
    return  spike_train_fixed

def collate_fn(samples):
    """Create a batch out of a list of tuple [(spike_train_tensor, str_label)]
    by zero-padding the spike trains"""
    #max_duration = max([s[0].shape[-1] for s in samples])
    max_duration = options.ts
    batch = torch.zeros(len(samples), n_features, max_duration)
    labels = []
    for i, s in enumerate(samples):
        batch[i, :, : s[0].shape[-1]] = s[0]
        labels.append(int(s[1].replace("z", "0").replace("o", "10")))
    return batch, torch.tensor(labels)

DATASET_PATH = '/home/rram/Nature-LSM/Ntdigist/data/n-tidigits.hdf5'
train_dataset = NTidigits(
    DATASET_PATH,
    is_train=True,
    transforms=rec_array_to_spike_train,
    only_single_digits=True,
)
test_dataset = NTidigits(
    DATASET_PATH,
    is_train=False,
    transforms=rec_array_to_spike_train,
    only_single_digits=True,
)

train_data_len = len(train_dataset)
print('train dataset lenth is',train_data_len)

test_data_len = len(test_dataset)
print('Test dataset lenth is',test_data_len)

print('=============N-TIDIGITS train data collection=============')
nt_tr_data_1 = []
nt_tr_data_2 = []
nt_tr_data_3 = []
nt_tr_data_4 = []
nt_tr_data_5 = []
nt_tr_data_6 = []
nt_tr_data_7 = []
nt_tr_data_8 = []
nt_tr_data_9 = []

nt_tr_label_1 = []
nt_tr_label_2 = []
nt_tr_label_3 = []
nt_tr_label_4 = []
nt_tr_label_5 = []
nt_tr_label_6 = []
nt_tr_label_7 = []
nt_tr_label_8 = []
nt_tr_label_9 = []

for i in range(len(train_dataset)):
    data = train_dataset[i][0]

    padding = torch.zeros(n_features, options.ts)
    padding[:data.shape[0],:data.shape[1]] = data
    # print('Image is',train_dataset[i][0])
    # print('Audio shape is', train_dataset[i][0].shape)
    # print('Audio shape is', padding.shape)
    # print('label is', train_dataset[i][1])
    labels = train_dataset[i][1]
    # print('label is', labels)

    if labels == "1":
        nt_tr_data_1.append(padding)
        nt_tr_label_1.append(torch.tensor(1.0))
    elif labels == "2":
        nt_tr_data_2.append(padding)
        nt_tr_label_2.append(torch.tensor(2.0))
    elif labels == "3":
        nt_tr_data_3.append(padding)
        nt_tr_label_3.append(torch.tensor(3.0))
    elif labels == "4":
        nt_tr_data_4.append(padding)
        nt_tr_label_4.append(torch.tensor(4.0))
    elif labels == "5":
        nt_tr_data_5.append(padding)
        nt_tr_label_5.append(torch.tensor(5.0))
    elif labels == "6":
        nt_tr_data_6.append(padding)
        nt_tr_label_6.append(torch.tensor(6.0))
    elif labels == "7":
        nt_tr_data_7.append(padding)
        nt_tr_label_7.append(torch.tensor(7.0))
    elif labels == "8":
        nt_tr_data_8.append(padding)
        nt_tr_label_8.append(torch.tensor(8.0))
    elif labels == "9":
        nt_tr_data_9.append(padding)
        nt_tr_label_9.append(torch.tensor(9.0))


print("nt data 1 lenth is", len(nt_tr_data_1))
print("nt data 2 lenth is", len(nt_tr_data_2))
print("nt data 3 lenth is", len(nt_tr_data_3))
print("nt data 4 lenth is", len(nt_tr_data_4))
print("nt data 5 lenth is", len(nt_tr_data_5))
print("nt data 6 lenth is", len(nt_tr_data_6))
print("nt data 7 lenth is", len(nt_tr_data_7))
print("nt data 8 lenth is", len(nt_tr_data_8))
print("nt data 9 lenth is", len(nt_tr_data_9))


print('=============N-TIDIGITS test data collection=============')
nt_te_data_0 = []
nt_te_data_1 = []
nt_te_data_2 = []
nt_te_data_3 = []
nt_te_data_4 = []
nt_te_data_5 = []
nt_te_data_6 = []
nt_te_data_7 = []
nt_te_data_8 = []
nt_te_data_9 = []

nt_te_label_1 = []
nt_te_label_2 = []
nt_te_label_3 = []
nt_te_label_4 = []
nt_te_label_5 = []
nt_te_label_6 = []
nt_te_label_7 = []
nt_te_label_8 = []
nt_te_label_9 = []

for i in range(len(test_dataset)):
    data = test_dataset[i][0]

    padding = torch.zeros(n_features, options.ts)
    padding[:data.shape[0],:data.shape[1]] = data
    # print('Image is',train_dataset[i][0])
    # print('Audio shape is', train_dataset[i][0].shape)
    # print('Audio shape is', padding.shape)
    # print('label is', train_dataset[i][1])
    labels = test_dataset[i][1]
    # print('label is', labels)

    if labels == "1":
        nt_te_data_1.append(padding)
        nt_te_label_1.append(torch.tensor(1.0))
    elif labels == "2":
        nt_te_data_2.append(padding)
        nt_te_label_2.append(torch.tensor(2.0))
    elif labels == "3":
        nt_te_data_3.append(padding)
        nt_te_label_3.append(torch.tensor(3.0))
    elif labels == "4":
        nt_te_data_4.append(padding)
        nt_te_label_4.append(torch.tensor(4.0))
    elif labels == "5":
        nt_te_data_5.append(padding)
        nt_te_label_5.append(torch.tensor(5.0))
    elif labels == "6":
        nt_te_data_6.append(padding)
        nt_te_label_6.append(torch.tensor(6.0))
    elif labels == "7":
        nt_te_data_7.append(padding)
        nt_te_label_7.append(torch.tensor(7.0))
    elif labels == "8":
        nt_te_data_8.append(padding)
        nt_te_label_8.append(torch.tensor(8.0))
    elif labels == "9":
        nt_te_data_9.append(padding)
        nt_te_label_9.append(torch.tensor(9.0))


print("nt test data 1 lenth is", len(nt_te_data_1))
print("nt test data 2 lenth is", len(nt_te_data_2))
print("nt test data 3 lenth is", len(nt_te_data_3))
print("nt test data 4 lenth is", len(nt_te_data_4))
print("nt test data 5 lenth is", len(nt_te_data_5))
print("nt test data 6 lenth is", len(nt_te_data_6))
print("nt test data 7 lenth is", len(nt_te_data_7))
print("nt test data 8 lenth is", len(nt_te_data_8))
print("nt test data 9 lenth is", len(nt_te_data_9))

##################################################
## train data
data_1_x = len(nm_tr_data_1)//len(nt_tr_data_1)
data_2_x = len(nm_tr_data_2)//len(nt_tr_data_2)
data_3_x = len(nm_tr_data_3)//len(nt_tr_data_3)
data_4_x = len(nm_tr_data_4)//len(nt_tr_data_4)
data_5_x = len(nm_tr_data_5)//len(nt_tr_data_5)
data_6_x = len(nm_tr_data_6)//len(nt_tr_data_6)
data_7_x = len(nm_tr_data_7)//len(nt_tr_data_7)
data_8_x = len(nm_tr_data_8)//len(nt_tr_data_8)
data_9_x = len(nm_tr_data_9)//len(nt_tr_data_9)

## test data
tedata_1_x = len(nm_te_data_1)//len(nt_te_data_1)
tedata_2_x = len(nm_te_data_2)//len(nt_te_data_2)
tedata_3_x = len(nm_te_data_3)//len(nt_te_data_3)
tedata_4_x = len(nm_te_data_4)//len(nt_te_data_4)
tedata_5_x = len(nm_te_data_5)//len(nt_te_data_5)
tedata_6_x = len(nm_te_data_6)//len(nt_te_data_6)
tedata_7_x = len(nm_te_data_7)//len(nt_te_data_7)
tedata_8_x = len(nm_te_data_8)//len(nt_te_data_8)
tedata_9_x = len(nm_te_data_9)//len(nt_te_data_9)

########################################### train data process ########################

train_img_all = nm_tr_data_1[:len(nt_tr_data_1)]  \
    + nm_tr_data_2[:len(nt_tr_data_2)]  \
    + nm_tr_data_3[:len(nt_tr_data_3)]  \
    + nm_tr_data_4[:len(nt_tr_data_4)]  \
    + nm_tr_data_5[:len(nt_tr_data_5)]  \
    + nm_tr_data_6[:len(nt_tr_data_6)]  \
    + nm_tr_data_7[:len(nt_tr_data_7)]  \
    # + nm_tr_data_8[:len(nt_tr_data_8)]  \
    # + nm_tr_data_9[:len(nt_tr_data_9)]

train_audio_all = nt_tr_data_1\
                  + nt_tr_data_2 \
                  + nt_tr_data_3 \
                  + nt_tr_data_4 \
                  + nt_tr_data_5 \
                  + nt_tr_data_6 \
                  + nt_tr_data_7 \
                  # + nt_tr_data_8 \
                  # + nt_tr_data_9 \


train_img_label = nm_tr_label_1[:len(nt_tr_label_1)] \
                + nm_tr_label_2[:len(nt_tr_label_2)] \
                + nm_tr_label_3[:len(nt_tr_label_3)] \
                + nm_tr_label_4[:len(nt_tr_label_4)] \
                + nm_tr_label_5[:len(nt_tr_label_5)] \
                + nm_tr_label_6[:len(nt_tr_label_6)] \
                + nm_tr_label_7[:len(nt_tr_label_7)] \
                # + nm_tr_label_8[:len(nt_tr_label_8)] \
                # + nm_tr_label_9[:len(nt_tr_label_9)] \


train_audio_label = nt_tr_label_1 \
                + nt_tr_label_2 \
                + nt_tr_label_3 \
                + nt_tr_label_4 \
                + nt_tr_label_5 \
                + nt_tr_label_6 \
                + nt_tr_label_7 \
                # + nt_tr_label_8 \
                # + nt_tr_label_9 \

print('train img len is', len(train_img_all))
print('train audio len is', len(train_audio_all))

print('train img label len is', len(train_img_label))
print('train audio label len is', len(train_audio_label))

train_img_all_data_list_tensor = torch.stack(train_img_all,0)
print(train_img_all_data_list_tensor.shape)

train_audio_all_list_tensor = torch.stack(train_audio_all,0)
print(train_audio_all_list_tensor.shape)

train_img_label_list_tensor = torch.stack(train_img_label,0)
print(train_img_label_list_tensor.shape)

train_audio_label_list_tensor = torch.stack(train_audio_label,0)
print(train_audio_label_list_tensor.shape)

########################################### (classes in train data) test data process ########################

nt_test_data_len = 100
print('test min len is', nt_test_data_len)
test_img_all_in = nm_te_data_1[:min(len(nt_te_data_1),nt_test_data_len)]  \
    + nm_te_data_2[:min(len(nt_te_data_2), nt_test_data_len)]  \
    + nm_te_data_3[:min(len(nt_te_data_3), nt_test_data_len)]  \
    + nm_te_data_4[:min(len(nt_te_data_4), nt_test_data_len)]  \
    + nm_te_data_5[:min(len(nt_te_data_5), nt_test_data_len)]  \
    + nm_te_data_6[:min(len(nt_te_data_6), nt_test_data_len)]  \
    + nm_te_data_7[:min(len(nt_te_data_7), nt_test_data_len)]

test_audio_all_in = nt_te_data_1[:min(len(nt_te_data_1), nt_test_data_len)]  \
                  + nt_te_data_2[:min(len(nt_te_data_2), nt_test_data_len)] \
                  + nt_te_data_3[:min(len(nt_te_data_3), nt_test_data_len)] \
                  + nt_te_data_4[:min(len(nt_te_data_4), nt_test_data_len)] \
                  + nt_te_data_5[:min(len(nt_te_data_5), nt_test_data_len)] \
                  + nt_te_data_6[:min(len(nt_te_data_6), nt_test_data_len)] \
                  + nt_te_data_7[:min(len(nt_te_data_7), nt_test_data_len)]

test_img_label_in = nm_te_label_1[:min(len(nt_te_label_1), nt_test_data_len)] \
                + nm_te_label_2[:min(len(nt_te_label_2), nt_test_data_len)] \
                + nm_te_label_3[:min(len(nt_te_label_3), nt_test_data_len)] \
                + nm_te_label_4[:min(len(nt_te_label_4), nt_test_data_len)] \
                + nm_te_label_5[:min(len(nt_te_label_5), nt_test_data_len)] \
                + nm_te_label_6[:min(len(nt_te_label_6), nt_test_data_len)] \
                + nm_te_label_7[:min(len(nt_te_label_7), nt_test_data_len)]


test_audio_label_in = nt_te_label_1[:min(len(nt_te_label_1), nt_test_data_len)] \
                + nt_te_label_2[:min(len(nt_te_label_2), nt_test_data_len)] \
                + nt_te_label_3[:min(len(nt_te_label_3), nt_test_data_len)] \
                + nt_te_label_4[:min(len(nt_te_label_4), nt_test_data_len)] \
                + nt_te_label_5[:min(len(nt_te_label_5), nt_test_data_len)] \
                + nt_te_label_6[:min(len(nt_te_label_6), nt_test_data_len)] \
                + nt_te_label_7[:min(len(nt_te_label_7), nt_test_data_len)]

print('test img len is', len(test_img_all_in))
print('test audio len is', len(test_audio_all_in))

print('test img label len is', len(test_img_label_in))
print('test audio label len is', len(test_audio_label_in))

test_img_all_data_list_tensor_in = torch.stack(test_img_all_in,0)
print(test_img_all_data_list_tensor_in.shape)

test_audio_all_list_tensor_in = torch.stack(test_audio_all_in,0)
print(test_audio_all_list_tensor_in.shape)

test_img_label_list_tensor_in = torch.stack(test_img_label_in,0)
print(test_img_label_list_tensor_in.shape)

test_audio_label_list_tensor_in = torch.stack(test_audio_label_in,0)
print(test_audio_label_list_tensor_in.shape)




########################################### (classes not in train data) test data process ########################

nt_test_data_len = 100
print('test min len is', nt_test_data_len)


test_img_all_out = nm_te_data_8[:min(len(nt_te_data_8), nt_test_data_len)]  \
    + nm_te_data_9[:min(len(nt_te_data_9), nt_test_data_len)]

test_audio_all_out = nt_te_data_8[:min(len(nt_te_data_8), nt_test_data_len)] \
                  + nt_te_data_9[:min(len(nt_te_data_9), nt_test_data_len)] \


test_img_label_out = nm_te_label_8[:min(len(nt_te_label_8), nt_test_data_len)] \
                + nm_te_label_9[:min(len(nt_te_label_9), nt_test_data_len)] \

test_audio_label_out = nt_te_label_8[:min(len(nt_te_label_8), nt_test_data_len)] \
                + nt_te_label_9[:min(len(nt_te_label_9), nt_test_data_len)] \

print('test img len is', len(test_img_all_out))
print('test audio len is', len(test_audio_all_out))

print('test img label len is', len(test_img_label_out))
print('test audio label len is', len(test_audio_label_out))

test_img_all_data_list_tensor_out = torch.stack(test_img_all_out,0)
print(test_img_all_data_list_tensor_out.shape)

test_audio_all_list_tensor_out = torch.stack(test_audio_all_out,0)
print(test_audio_all_list_tensor_out.shape)

test_img_label_list_tensor_out = torch.stack(test_img_label_out,0)
print(test_img_label_list_tensor_out.shape)

test_audio_label_list_tensor_out = torch.stack(test_audio_label_out,0)
print(test_audio_label_list_tensor_out.shape)

#######################################
########## Data loader ############
train_dataset_normal = CustomTensorDataset(
    tensors=(train_img_all_data_list_tensor, train_audio_all_list_tensor,
             train_img_label_list_tensor, train_audio_label_list_tensor),
    transform=None)

test_dataset_normal_in = CustomTensorDataset(
    tensors=(test_img_all_data_list_tensor_in, test_audio_all_list_tensor_in,
             test_img_label_list_tensor_in, test_audio_label_list_tensor_in),
    transform=None)

test_dataset_normal_out = CustomTensorDataset(
    tensors=(test_img_all_data_list_tensor_out, test_audio_all_list_tensor_out,
             test_img_label_list_tensor_out, test_audio_label_list_tensor_out),
    transform=None)


options_batch = options.batch
train_loader_crop = torch.utils.data.DataLoader(train_dataset_normal,
                                                batch_size=options_batch,
                                                shuffle=True)
test_loader_crop_in = torch.utils.data.DataLoader(test_dataset_normal_in,
                                                batch_size=options_batch,
                                                shuffle=False)
test_loader_crop_out = torch.utils.data.DataLoader(test_dataset_normal_out,
                                                batch_size=options_batch,
                                                shuffle=False)

img_encoder = NM_simulation_Gau_LSM_Model(cfg=[16 * 16 * 1, 200, 9], time_window = 50, lens = 0.25, thresh = options.img_vth,
                                          decay_lsm=options.img_decay, const=options.img_const, scale=100.0)

aud_encoder = NT_simulation_Gau_LSM_Model(cfg=[64, 200, 9], time_window = 129, lens = 0.25, thresh = options.aud_vth,
                                          decay_lsm=options.aud_decay, const=options.aud_const, scale=100.0)
img_encoder.to(device)
aud_encoder.to(device)

###################### LSM encoder #############################
print("=============== LSM Encoding for train dataset =================")

counter_img_train_data_list = []
counter_img_train_label_list = []

counter_aud_train_data_list = []
counter_aud_train_label_list = []

for i, (images, audios, img_labels, aud_labels) in enumerate(train_loader_crop):

    images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
    audios = Variable(audios.float(), requires_grad=False).to(device)

    img_fea = img_encoder(images0)
    aud_fea = aud_encoder(audios)

    counter_img_train_data_list.append(img_fea)
    counter_img_train_label_list.append(img_labels)

    counter_aud_train_data_list.append(aud_fea)
    counter_aud_train_label_list.append(aud_labels)

counter_img_train_data_list_tensor = torch.cat(counter_img_train_data_list,0)
print(counter_img_train_data_list_tensor.shape)
counter_img_train_label_list_tensor = torch.cat(counter_img_train_label_list,0)
print(counter_img_train_label_list_tensor.shape)

counter_aud_train_data_list_tensor = torch.cat(counter_aud_train_data_list,0)
print(counter_aud_train_data_list_tensor.shape)
counter_aud_train_label_list_tensor = torch.cat(counter_aud_train_label_list,0)
print(counter_aud_train_label_list_tensor.shape)

print("=============== LSM Encoding for test dataset (in classes) =================")

in_counter_img_test_data_list = []
in_counter_img_test_label_list = []

in_counter_aud_test_data_list = []
in_counter_aud_test_label_list = []

for i, (images, audios, img_labels, aud_labels) in enumerate(test_loader_crop_in):

    images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
    audios = Variable(audios.float(), requires_grad=False).to(device)

    img_fea = img_encoder(images0)
    aud_fea = aud_encoder(audios)

    in_counter_img_test_data_list.append(img_fea)
    in_counter_img_test_label_list.append(img_labels)

    in_counter_aud_test_data_list.append(aud_fea)
    in_counter_aud_test_label_list.append(aud_labels)


in_counter_img_test_data_list_tensor = torch.cat(in_counter_img_test_data_list,0)
print(in_counter_img_test_data_list_tensor.shape)
in_counter_img_test_label_list_tensor = torch.cat(in_counter_img_test_label_list,0)
print(in_counter_img_test_label_list_tensor.shape)

in_counter_aud_test_data_list_tensor = torch.cat(in_counter_aud_test_data_list,0)
print(in_counter_aud_test_data_list_tensor.shape)
in_counter_aud_test_label_list_tensor = torch.cat(in_counter_aud_test_label_list,0)
print(in_counter_aud_test_label_list_tensor.shape)



print("=============== LSM Encoding for test dataset (out classes) =================")

out_counter_img_test_data_list = []
out_counter_img_test_label_list = []

out_counter_aud_test_data_list = []
out_counter_aud_test_label_list = []

for i, (images, audios, img_labels, aud_labels) in enumerate(test_loader_crop_out):

    images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
    audios = Variable(audios.float(), requires_grad=False).to(device)

    img_fea = img_encoder(images0)
    aud_fea = aud_encoder(audios)

    out_counter_img_test_data_list.append(img_fea)
    out_counter_img_test_label_list.append(img_labels)

    out_counter_aud_test_data_list.append(aud_fea)
    out_counter_aud_test_label_list.append(aud_labels)


out_counter_img_test_data_list_tensor = torch.cat(out_counter_img_test_data_list,0)
print(out_counter_img_test_data_list_tensor.shape)
out_counter_img_test_label_list_tensor = torch.cat(out_counter_img_test_label_list,0)
print(out_counter_img_test_label_list_tensor.shape)

out_counter_aud_test_data_list_tensor = torch.cat(out_counter_aud_test_data_list,0)
print(out_counter_aud_test_data_list_tensor.shape)
out_counter_aud_test_label_list_tensor = torch.cat(out_counter_aud_test_label_list,0)
print(out_counter_aud_test_label_list_tensor.shape)

print('==============================Begin train by the Readout ANN===========================================')

########## Data loader ############
train_dataset_feature = CustomTensorDataset(
    tensors=(counter_img_train_data_list_tensor, counter_aud_train_data_list_tensor,
             counter_img_train_label_list_tensor, counter_aud_train_label_list_tensor),
    transform=None)

in_test_dataset_feature = CustomTensorDataset(
    tensors=(in_counter_img_test_data_list_tensor, in_counter_aud_test_data_list_tensor,
             in_counter_img_test_label_list_tensor, in_counter_aud_test_label_list_tensor),
    transform=None)

out_test_dataset_feature = CustomTensorDataset(
    tensors=(out_counter_img_test_data_list_tensor, out_counter_aud_test_data_list_tensor,
             out_counter_img_test_label_list_tensor, out_counter_aud_test_label_list_tensor),
    transform=None)


options_feature_batch = options.feature_batch

train_feature_loader_crop = torch.utils.data.DataLoader(train_dataset_feature,
                                                batch_size=options_feature_batch,
                                                shuffle=True)

in_test_feature_loader_crop = torch.utils.data.DataLoader(in_test_dataset_feature,
                                                batch_size=options_feature_batch,
                                                shuffle=False)

out_test_feature_loader_crop = torch.utils.data.DataLoader(out_test_dataset_feature,
                                                batch_size=options_feature_batch,
                                                shuffle=False)


if options.models == 'OneFC':
    net = Contrastive_Simple(project_dim=options.pd, act_fun=options.act)
elif options.models == 'TwoFC':
    net = Contrastive_twolayers(project_dim=options.pd, act_fun=options.act)
elif options.models == 'ThreeFC':
    net = Contrastive_threelayers(project_dim=options.pd, act_fun=options.act)


net.to(device)


options_lr = options.lr
optimizer = torch.optim.Adam(net.parameters(), lr=options_lr)

options_epochs = options.epochs
print('Feature batch size is',options_feature_batch)


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    for i, (f_images, f_audios, img_labels, aud_labels) in enumerate(valid_loader):

        images0 = Variable(f_images, requires_grad=True).to(device)
        audios = Variable(f_audios, requires_grad=False).to(device)

        loss = model(images0, audios)
        count = images0.size(0)
        loss_meter.update(loss.item(), count)

    return loss_meter


def valid_acc(model,
              counter_img_test_data_list_tensor,
              counter_img_test_label_list_tensor,
              counter_aud_test_data_list_tensor,
              counter_aud_test_label_list_tensor,
              top=1):

    # image_embeddings = model.image_projection(counter_img_test_data_list_tensor)
    # audio_embeddings = model.audio_projection(counter_aud_test_data_list_tensor)

    image_embeddings_n = model.image_projection(counter_img_test_data_list_tensor)
    audio_embeddings_n = model.audio_projection(counter_aud_test_data_list_tensor)


    # image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    # audio_embeddings_n = F.normalize(audio_embeddings, p=2, dim=-1)

    ### audio search image #########
    count_m = 0
    for idx in range(len(counter_aud_test_label_list_tensor)):
        img_label = counter_img_test_label_list_tensor[idx]

        dot_similarity = audio_embeddings_n[idx] @ image_embeddings_n.T

        values, indices = torch.topk(dot_similarity.squeeze(0), top)
        pre = counter_img_test_label_list_tensor[indices.cpu()]

        if img_label.cpu() in pre:
            count_m = count_m + 1
        # else:
        #     # print("no")
    top_acc_a2i = count_m/len(counter_aud_test_label_list_tensor)*100.0

    ### image search audio ########
    count_aud = 0
    for idx in range(len(counter_img_test_label_list_tensor)):
        aud_label = counter_aud_test_label_list_tensor[idx]

        dot_similarity = image_embeddings_n[idx] @ audio_embeddings_n.T

        values, indices = torch.topk(dot_similarity.squeeze(0), top)
        pre = counter_aud_test_label_list_tensor[indices.cpu()]

        if aud_label.cpu() in pre:
            # print("yes")
            count_aud = count_aud + 1
        # else:
        #     # print("no")
    top_acc_i2a = count_aud / len(counter_img_test_label_list_tensor) * 100.0

    return top_acc_a2i, top_acc_i2a



best_acc_top1_a2i = 0.0
best_acc_top1_i2a = 0.0

cur_out_acc_a2i = 0.0
cur_out_acc_i2a = 0.0

for epoch in range(options_epochs):
    print(f"Epoch: {epoch + 1}")

    net.train()
    loss_meter = AvgMeter()
    for i, (f_images, f_audios, img_labels, aud_labels) in enumerate(train_feature_loader_crop):
        net.zero_grad()
        optimizer.zero_grad()
        f_images = Variable(f_images, requires_grad=True).to(device)
        f_audios = Variable(f_audios, requires_grad=False).to(device)

        loss = net(f_images,f_audios)

        count = f_images.size(0)
        loss_meter.update(loss.item(), count)
        loss.backward()
        optimizer.step()
        # tqdm_object.set_postfix(train_loss=loss_meter.avg)

    print(f"Train Loss is: {loss_meter.avg}")


    net.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(net, in_test_feature_loader_crop)
        print(f"Test Loss is: {valid_loss.avg}")

        in_acc_top1_a2i, in_acc_top1_i2a = valid_acc(net,
                                               in_counter_img_test_data_list_tensor,
                                               in_counter_img_test_label_list_tensor,
                                               in_counter_aud_test_data_list_tensor,
                                               in_counter_aud_test_label_list_tensor,
                                               top=1)
        print(f"Top1 Accuracy for audio search image (in-classes): {in_acc_top1_a2i}")
        print(f"Top1 Accuracy for image search audio (in-classes): {in_acc_top1_i2a}")


        out_acc_top1_a2i, out_acc_top1_i2a = valid_acc(net,
                                               out_counter_img_test_data_list_tensor,
                                               out_counter_img_test_label_list_tensor,
                                               out_counter_aud_test_data_list_tensor,
                                               out_counter_aud_test_label_list_tensor,
                                               top=1)
        print(f"Top1 Accuracy for audio search image (out-classes): {out_acc_top1_a2i}")
        print(f"Top1 Accuracy for image search audio (out-classes): {out_acc_top1_i2a}")


    ### top1
    if in_acc_top1_a2i >= 65 and out_acc_top1_a2i>=85:
    # if in_acc_top1_a2i >= 0 and out_acc_top1_a2i >= 0:
        if in_acc_top1_a2i > best_acc_top1_a2i:
            best_acc_top1_a2i = in_acc_top1_a2i
            cur_out_acc_a2i = out_acc_top1_a2i

            state = {
                'img_encoder': img_encoder.state_dict(),
                'aud_encoder': aud_encoder.state_dict(),
                'clip_net': net.state_dict(),
                'best_acc': best_acc_top1_a2i,
                'out_acc': cur_out_acc_a2i
            }
            torch.save(state, f'{names}_top1_a2i.t7')
            print("Saved Best Model!")

            print('In-classes audio best accuracy is', best_acc_top1_a2i)
            print('Out-classes audio best accuracy is', cur_out_acc_a2i)


    ### top1
    if in_acc_top1_i2a >= 70 and out_acc_top1_i2a >=60:
    # if in_acc_top1_i2a >= 0 and out_acc_top1_i2a >= 0:
        if in_acc_top1_i2a > best_acc_top1_i2a:
            best_acc_top1_i2a = in_acc_top1_i2a
            cur_out_acc_i2a = out_acc_top1_i2a

            state = {
                'img_encoder': img_encoder.state_dict(),
                'aud_encoder': aud_encoder.state_dict(),
                'clip_net': net.state_dict(),
                'best_acc': best_acc_top1_i2a,
                'out_acc': cur_out_acc_i2a
            }
            torch.save(state, f'{names}_top1_i2a.t7')
            print("Saved Best Model!")

            print('In-classes audio best accuracy is', best_acc_top1_i2a)
            print('Out-classes audio best accuracy is', cur_out_acc_i2a)

print(f"Finally Best Top1-acc for audio search images: {best_acc_top1_a2i}")
print(f"Finally Best Top1-acc for image search audios: {best_acc_top1_i2a}")
print(f"Finally Out Top1-acc for audio search images: {cur_out_acc_a2i}")
print(f"Finally Out Top1-acc for image search audios: {cur_out_acc_i2a}")




