import torch
import time
from datetime import datetime
import os
import argparse
from utils import MyCenterCropDataset
from utils import AvgMeter
from torch.autograd import Variable
from utils import CustomTensorDataset_pro
from ebdataset.audio import NTidigits
import random
import numpy as np
from quantities import ms, second
from models import PrototypicalNetworks_ete
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1568,
                        help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate ')
 

    parser.add_argument('--pd', type=int, default=2048,
                        help='projection dimention')
    parser.add_argument('--tp', type=float, default=1.0,
                        help='temperature factor for cosine similarities')

    parser.add_argument('--img_const', type=float, default=0.3,
                        help='image encoder weight constant')
    parser.add_argument('--aud_const', type=float, default=0.8,
                        help='audio encoder weight constant')

    parser.add_argument('--img_decay', type=float, default=0.9,
                        help='image encoder decay')
    parser.add_argument('--aud_decay', type=float, default=0.97,
                        help='audio encoder decay')

    parser.add_argument('--img_vth', type=float, default=5.0,
                        help='image vth')
    parser.add_argument('--aud_vth', type=float, default=5.3,
                        help='image vth')

    parser.add_argument('--feature_batch', type=int, default=1568,
                        help='encoding feature batch size')
    parser.add_argument('--ts', type=int, default=129, 
                        help='time_step')
    parser.add_argument('--Dt', type=int, default=1, 
                        help='Integration window')

    parser.add_argument('--feat_dim', type=int, default=200, )


    ## configurations
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')


    args = parser.parse_args()
    return args


options = parse_args()
print(options)
# print time
print('time: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

SAVE_DIR = 'zero-shot-prototype'


####set seed###
setup_seed(options.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR,
f'seed{options.seed}_audvth{options.aud_vth}_imgvth{options.img_vth}_audconst{options.aud_const}_imgconst{options.img_const}_auddecay{options.aud_decay}_imgdecay{options.img_decay}_temperature{options.tp}_projectiondimention{options.pd}_featurebatch{options.feature_batch}_batch{options.batch}_lr{options.lr}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')
device = torch.device('cuda:0' if (torch.cuda.is_available() and options.cuda !='cpu') else "cpu")

####################### NMNIST Datasets ########################
train_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_train_data.mat'
test_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_test_data.mat'
train_dataset_crop = MyCenterCropDataset(
    train_path, 'nmnist_h', new_width=16, new_height=16)
# test data size: 10k
test_dataset_crop = MyCenterCropDataset(
    test_path, 'nmnist_r', new_width=16, new_height=16)


def valid_acc(model, img_que_v, aud_que_v, que_img_label, que_aud_label, top=1):

    image_embeddings_n = F.normalize(img_que_v, p=2, dim=-1)
    audio_embeddings_n = F.normalize(aud_que_v, p=2, dim=-1)

    ### audio search image #########
    euc_dists = torch.cdist(audio_embeddings_n, image_embeddings_n, p=2)
    indices = torch.argmin(euc_dists, dim=1)
    img_label = que_img_label[indices.cpu()]
    correct = (img_label == que_aud_label).sum().item()
    top_acc_a2i = correct / len(que_aud_label) * 100.0

    ### image search audio ########
    euc_dists = torch.cdist(image_embeddings_n, audio_embeddings_n, p=2)
    indices = torch.argmin(euc_dists, dim=1)
    aud_label = que_aud_label[indices.cpu()]
    correct = (aud_label == que_img_label).sum().item()
    top_acc_i2a = correct / len(que_img_label) * 100.0

    return top_acc_a2i, top_acc_i2a

# print('=============NMNIST train data collection=============')
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
    labels = torch.argmax(train_dataset_crop[i][1],dim=0)
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
# print('=============NMNIST test data collection=============')
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
    labels = torch.argmax(test_dataset_crop[i][1],dim=0)

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
# print('train dataset lenth is',train_data_len)

test_data_len = len(test_dataset)
# print('Test dataset lenth is',test_data_len)

# print('=============N-TIDIGITS train data collection=============')
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
    labels = train_dataset[i][1]

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

# print('=============N-TIDIGITS test data collection=============')
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

########################################### train data process ########################

tr_sup_img = nm_tr_data_1[:len(nt_tr_data_1)]  \
    + nm_tr_data_2[:len(nt_tr_data_2)]  \
    + nm_tr_data_3[:len(nt_tr_data_3)]  \
    + nm_tr_data_4[:len(nt_tr_data_4)]  \
    + nm_tr_data_5[:len(nt_tr_data_5)]  \
    + nm_tr_data_6[:len(nt_tr_data_6)]  \
    + nm_tr_data_7[:len(nt_tr_data_7)]  
tr_sup_audio = nt_tr_data_1\
                  + nt_tr_data_2 \
                  + nt_tr_data_3 \
                  + nt_tr_data_4 \
                  + nt_tr_data_5 \
                  + nt_tr_data_6 \
                  + nt_tr_data_7
tr_sup_img_label = nm_tr_label_1[:len(nt_tr_label_1)] \
                + nm_tr_label_2[:len(nt_tr_label_2)] \
                + nm_tr_label_3[:len(nt_tr_label_3)] \
                + nm_tr_label_4[:len(nt_tr_label_4)] \
                + nm_tr_label_5[:len(nt_tr_label_5)] \
                + nm_tr_label_6[:len(nt_tr_label_6)] \
                + nm_tr_label_7[:len(nt_tr_label_7)] 
tr_sup_aud_label = nt_tr_label_1 \
                + nt_tr_label_2 \
                + nt_tr_label_3 \
                + nt_tr_label_4 \
                + nt_tr_label_5 \
                + nt_tr_label_6 \
                + nt_tr_label_7 
# training query data
tr_que_img = nm_te_data_1[:len(nt_tr_data_1)]  \
    + nm_te_data_2[:len(nt_tr_data_2)]  \
    + nm_te_data_3[:len(nt_tr_data_3)]  \
    + nm_te_data_4[:len(nt_tr_data_4)]  \
    + nm_te_data_5[:len(nt_tr_data_5)]  \
    + nm_te_data_6[:len(nt_tr_data_6)]  \
    + nm_te_data_7[:len(nt_tr_data_7)]  
tr_que_aud =nt_te_data_1[:len(nt_tr_data_1)]\
                  + nt_te_data_2[:len(nt_tr_data_2)] \
                  + nt_te_data_3[:len(nt_tr_data_3)] \
                  + nt_te_data_4[:len(nt_tr_data_4)] \
                  + nt_te_data_5[:len(nt_tr_data_5)] \
                  + nt_te_data_6[:len(nt_tr_data_6)] \
                  + nt_te_data_7[:len(nt_tr_data_7)] 
tr_que_img_label = nm_te_label_1[:len(nt_tr_label_1)] \
                + nm_te_label_2[:len(nt_tr_label_2)] \
                + nm_te_label_3[:len(nt_tr_label_3)] \
                + nm_te_label_4[:len(nt_tr_label_4)] \
                + nm_te_label_5[:len(nt_tr_label_5)] \
                + nm_te_label_6[:len(nt_tr_label_6)] \
                + nm_te_label_7[:len(nt_tr_label_7)] 
tr_que_audio_label = nt_te_label_1[:len(nt_tr_data_1)] \
                + nt_te_label_2[:len(nt_tr_data_2)] \
                + nt_te_label_3[:len(nt_tr_data_3)] \
                + nt_te_label_4[:len(nt_tr_data_4)] \
                + nt_te_label_5[:len(nt_tr_data_5)] \
                + nt_te_label_6[:len(nt_tr_data_6)] \
                + nt_te_label_7[:len(nt_tr_data_7)] 

tr_sup_img = torch.stack(tr_sup_img,0)
tr_sup_aud = torch.stack(tr_sup_audio,0)

tr_sup_img_label = torch.stack(tr_sup_img_label,0)
tr_sup_aud_label = torch.stack(tr_sup_aud_label,0)

tr_que_img = torch.stack(tr_que_img, 0)
tr_que_aud = torch.stack(tr_que_aud, 0)
tr_que_img_label = torch.stack(tr_que_img_label, 0)
tr_que_aud_label = torch.stack(tr_que_audio_label, 0)
########################################### test data process ########################

# 100 samples per class for test
# nt_test_data_len = 100
# print('test min len is', nt_test_data_len)


te_que_img = nm_te_data_8[:min(len(nt_te_data_8), len(nt_tr_data_8))]  \
    + nm_te_data_9[:min(len(nt_te_data_9), len(nt_tr_data_9))]

te_que_aud = nt_te_data_8[:min(len(nt_te_data_8), len(nt_tr_data_8))] \
                  + nt_te_data_9[:min(len(nt_te_data_9), len(nt_tr_data_9))] \

te_que_img_label = nm_te_label_8[:min(len(nt_te_label_8), len(nt_tr_label_8))] \
                + nm_te_label_9[:min(len(nt_te_label_9), len(nt_tr_label_9))] \

te_que_aud_label = nt_te_label_8[:min(len(nt_te_label_8), len(nt_tr_label_8))] \
                + nt_te_label_9[:min(len(nt_te_label_9), len(nt_tr_label_9))] 

te_sup_img = nm_tr_data_8[:min(len(nt_tr_data_8), len(nt_tr_label_8))]  \
    + nm_tr_data_9[:min(len(nt_tr_data_9), len(nt_tr_label_9))]

te_sup_audio = nt_tr_data_8[:min(len(nt_tr_data_8), len(nt_tr_label_8))] \
                  + nt_tr_data_9[:min(len(nt_tr_data_9), len(nt_tr_label_9))] \

te_sup_img_label = nm_tr_label_8[:min(len(nt_tr_label_8), len(nt_tr_label_8))] \
                + nm_tr_label_9[:min(len(nt_tr_label_9), len(nt_tr_label_9))] \

te_sup_aud_label = nt_tr_label_8[:min(len(nt_tr_label_8), len(nt_tr_label_8))] \
                + nt_tr_label_9[:min(len(nt_tr_label_9), len(nt_tr_label_9))] 

te_que_img = torch.stack(te_que_img,0)
te_que_aud = torch.stack(te_que_aud,0)
te_que_img_label = torch.stack(te_que_img_label,0)
te_que_aud_label = torch.stack(te_que_aud_label,0)

te_sup_img = torch.stack(te_sup_img, 0)
te_sup_aud = torch.stack(te_sup_audio, 0)
te_sup_img_label = torch.stack(te_sup_img_label, 0)
te_sup_aud_label = torch.stack(te_sup_aud_label, 0)
tr_sup_img_label, tr_sup_aud_label, tr_que_img_label, tr_que_aud_label = tr_sup_img_label.to(device, torch.int64), tr_sup_aud_label.to(device, torch.int64), tr_que_img_label.to(device, torch.int64), tr_que_aud_label.to(device, torch.int64)
te_sup_img_label, te_sup_aud_label, te_que_img_label, te_que_aud_label = te_sup_img_label.to(device, torch.int64), te_sup_aud_label.to(device, torch.int64), te_que_img_label.to(device, torch.int64), te_que_aud_label.to(device, torch.int64)
tr_sup_img_label, tr_sup_aud_label, tr_que_img_label, tr_que_aud_label = tr_sup_img_label - 1, tr_sup_aud_label - 1, tr_que_img_label - 1, tr_que_aud_label - 1
te_sup_img_label, te_sup_aud_label, te_que_img_label, te_que_aud_label = te_sup_img_label - 8, te_sup_aud_label - 8, te_que_img_label - 8, te_que_aud_label - 8
#######################################
########## Data loader ############
train_dataset = CustomTensorDataset_pro(
    tensors=(tr_sup_img, tr_sup_aud,
             tr_sup_img_label, tr_sup_aud_label, tr_que_img, tr_que_aud, tr_que_img_label, tr_que_aud_label),
    transform=None)

test_dataset = CustomTensorDataset_pro(
    tensors=(te_sup_img, te_sup_aud, te_sup_img_label, te_sup_aud_label, te_que_img, te_que_aud, te_que_img_label, te_que_aud_label),
    transform=None)

options_batch = options.batch
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=options_batch,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=options_batch,
                                               shuffle=False)


backbone = torch.nn.ModuleDict({"img_projection": torch.nn.Linear(options.feat_dim, options.pd),
                               "aud_projection": torch.nn.Linear(options.feat_dim, options.pd)})
net = PrototypicalNetworks_ete(backbone, options.feat_dim,
                               options.img_vth, options.img_decay, options.img_const,
                               options.aud_vth, options.aud_decay, options.aud_const,
                               device=device)

criterion = torch.nn.CrossEntropyLoss()
options_lr = options.lr
optimizer = torch.optim.Adam(net.parameters(), lr=options_lr)

num_epochs = options.epochs
# print('Feature batch size is',options_feature_batch)


best_acc_top1_a2i = 0.0
best_acc_top1_i2a = 0.0

cur_out_acc_a2i = 0.0
cur_out_acc_i2a = 0.0

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    net.train()
    correction = 0
    loss_meter = AvgMeter()
    for i, (sup_img, sup_aud, sup_img_label, sup_aud_label, que_img, que_aud, que_img_label, que_aud_label) in enumerate(train_loader):

        net.zero_grad()
        optimizer.zero_grad()
        sup_img = Variable(sup_img[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
        sup_aud = Variable(sup_aud.float(), requires_grad=False).to(device)
        que_img = Variable(que_img[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
        que_aud = Variable(que_aud.float(), requires_grad=False).to(device)

        tr_score, tr_proto, tr_img_que_v, tr_aud_que_v = net(sup_img, sup_aud, sup_img_label, sup_aud_label, que_img, que_aud)
        que_label = torch.cat([que_img_label, que_aud_label], dim=0).to(device, torch.int64)
        # print(f'max label is {que_label.max()}')
        correction += (tr_score.argmax(dim=1) == que_label).sum().item()
        count = sup_img.size(0)
        loss = criterion(tr_score, que_label)
        loss_meter.update(loss.item(), count)
        try:
            loss.backward()
        except RuntimeError:
            print(f'Error: max label is {que_label.max()}')
            continue
        optimizer.step()

    with torch.no_grad():
        net.eval()
    # print("=============== LSM Encoding for test dataset =================")
        for i, (sup_img, sup_aud, sup_img_label, sup_aud_label, que_img, que_aud, que_img_label, que_aud_label) in enumerate(test_loader):

            sup_img = Variable(sup_img[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
            sup_aud = Variable(sup_aud.float(), requires_grad=False).to(device)
            que_img = Variable(que_img[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
            que_aud = Variable(que_aud.float(), requires_grad=False).to(device)

            te_score, te_proto, te_img_que_v, te_aud_que_v = net(sup_img, sup_aud, sup_img_label, sup_aud_label, que_img, que_aud)
            # correction
            correction += (tr_score.argmax(dim=1) == que_label).sum().item()
            count = sup_img.size(0)


        tr_acc_a2i, tr_acc_i2a = valid_acc(net, tr_img_que_v, tr_aud_que_v, tr_que_img_label, tr_que_aud_label, top=1)
        print(f"Top1 Accuracy for audio search image (in-classes): {tr_acc_a2i}")
        print(f"Top1 Accuracy for image search audio (in-classes): {tr_acc_i2a}")

        acc_a2i, acc_top1_i2a = valid_acc(net, te_img_que_v, te_aud_que_v, te_que_img_label, te_que_aud_label, top=1)
        print(f"Top1 Accuracy for audio search image (out-classes): {acc_a2i}")
        print(f"Top1 Accuracy for image search audio (out-classes): {acc_top1_i2a}")


        ### top1
        if tr_acc_a2i >= 50 and acc_a2i>=50:
        # if in_acc_top1_a2i >= 0 and out_acc_top1_a2i >= 0:
            if tr_acc_a2i > best_acc_top1_a2i:
                best_acc_top1_a2i = tr_acc_a2i
                cur_out_acc_a2i = acc_a2i

                print('In-classes audio best accuracy is', best_acc_top1_a2i)
                print('Out-classes audio best accuracy is', cur_out_acc_a2i)

                ### top1
        if tr_acc_i2a >= 50 and acc_top1_i2a>=50:
        # if in_acc_top1_a2i >= 0 and out_acc_top1_a2i >= 0:
            if tr_acc_i2a > best_acc_top1_i2a:
                best_acc_top1_i2a = tr_acc_i2a
                cur_out_acc_i2a = acc_top1_i2a

                print('In-classes image best accuracy is', best_acc_top1_i2a)
                print('Out-classes image best accuracy is', cur_out_acc_i2a)

print(f"Finally Best In Top1-acc for audio search images: {best_acc_top1_a2i}")
print(f"Finally Best Out Top1-acc for audio search images: {cur_out_acc_a2i}")
print(f"Finally Best In Top1-acc for image search audio: {best_acc_top1_i2a}")
print(f"Finally Best Out Top1-acc for image search audio: {cur_out_acc_i2a}")