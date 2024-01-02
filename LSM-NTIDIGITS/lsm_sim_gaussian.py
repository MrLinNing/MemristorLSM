import torch
import time
from datetime import datetime
import os
import argparse
import torch.nn as nn
from torch.autograd import Variable
from ebdataset.audio import NTidigits
from quantities import ms, second
import numpy as np
import torch.nn.functional as F
import random

from utils import CustomTensorDataset
from models import simulation_Gau_LSM_Model
from models import Read_ANN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    ## hyper-parameters
    parser.add_argument('--batch', type=int, default=1000, metavar='B',
                        help='input batch size for training default (128)')
    parser.add_argument('--ann_batch', type=int, default=200, metavar='B',
                        help='input batch size for training default (128)')
    parser.add_argument('--epochs', type=int, default=600, metavar='E',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate ')

    ## model informations
    parser.add_argument('--scale', type=float, default=100.0, metavar='G',
                        help='scale factor (default: 100.0)')
    parser.add_argument('--Decay_lsm', type=float, default=0.9, metavar='D_L',
                        help='Decay Value of LIF for LSM')
    parser.add_argument('--Vth', type=float, default=50, metavar='V',
                        help='Vth value of LIF')
    parser.add_argument('--Tw', type=int, default=1469, metavar='T',
                        help='Time window value of LIF')
    parser.add_argument('--ts', type=int, default=50, metavar='T',
                        help='time_step')
    parser.add_argument('--const', type=float, default=1.0, metavar='V',
                        help='Vth value of LIF')
    parser.add_argument('--Dt', type=int, default=1, metavar='T',
                        help='Integration window')
    parser.add_argument('--ARCHI', type=str, default="64,256,11", metavar='A',
                        help='Network node of SNN')

    ## training configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='random seed')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

####set seed###
setup_seed(options.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)

SAVE_DIR = 'Sim_LSM_Submit'

cfg_net = [int(item) for item in options.ARCHI.split(',')]
print(cfg_net)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR,
                        f'seed{options.seed}_b{options.ann_batch}_Dt{options.Dt}_TimeStep{options.ts}_Timewindow{options.Tw}_decaylsm{options.Decay_lsm}_lr{options.lr}_ARCHI{cfg_net}_Vth{options.Vth}_const{options.const}_scale{options.scale}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')

device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")


######
# N-Tidigits
######
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

data_loader_parameters = {
    "batch_size": options.batch,
    "pin_memory": True,
    "collate_fn": collate_fn,
}

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                shuffle=True,
                                                **data_loader_parameters)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               shuffle=False,
                                               **data_loader_parameters)


net = simulation_Gau_LSM_Model(cfg=cfg_net, time_window=options.Tw, lens=0.25, thresh=options.Vth,
                               decay_lsm=options.Decay_lsm, const=options.const,scale=options.scale)
net.to(device)

print('==============================Begin process by the LSM========================================')
################# process by the RRAM-LSM ###################

### train data###
counter_train_data_list = []
counter_train_label_list = []
for i, (images, labels) in enumerate(train_loader):
    net.zero_grad()
    images0 = Variable(images.float(), requires_grad=True).to(device)
    outputs = net(images0)
    counter_train_data_list.append(outputs)
    counter_train_label_list.append(labels)

counter_train_data_list_tensor = torch.cat(counter_train_data_list,0)
print(counter_train_data_list_tensor.shape)

counter_train_label_list_tensor = torch.cat(counter_train_label_list,0)
print(counter_train_label_list_tensor.shape)


### test data ####
counter_val_data_list = []
counter_val_label_list = []

for i, (images, labels) in enumerate(test_loader):
    net.zero_grad()
    images0 = Variable(images.float(), requires_grad=True).to(device)
    outputs = net(images0)
    counter_val_data_list.append(outputs)
    counter_val_label_list.append(labels)


counter_val_data_list_tensor = torch.cat(counter_val_data_list,0)
print(counter_val_data_list_tensor.shape)

counter_val_label_list_tensor = torch.cat(counter_val_label_list,0)
print(counter_val_label_list_tensor.shape)

####################

print('==============================Begin train by the Readout ANN===========================================')

########## train by Readout ANN############
train_dataset_normal = CustomTensorDataset(
    tensors=(counter_train_data_list_tensor, counter_train_label_list_tensor),
    transform=None)

test_dataset_normal = CustomTensorDataset(
    tensors=(counter_val_data_list_tensor, counter_val_label_list_tensor),
    transform=None)

options_batch = options.ann_batch

train_loader_crop = torch.utils.data.DataLoader(train_dataset_normal,
                                          batch_size=options_batch)
test_loader_crop = torch.utils.data.DataLoader(test_dataset_normal,
                                                batch_size=options_batch)

net2 = Read_ANN(cfg=cfg_net)
net2.to(device)

options_lr = options.lr

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net2.parameters(), lr=options_lr)

options_epochs = options.epochs

best_acc = 0
tr_accs, te_accs, tr_losses, te_losses = [], [], [], []
for epoch in range(options_epochs):
    running_loss, te_loss_epoch = 0, 0
    start_time = time.time()
    tr_acc_batches = []
    for i, (images, labels) in enumerate(train_loader_crop):
        net.zero_grad()
        optimizer.zero_grad()
        images0 = Variable(images.float(), requires_grad=True).to(device)
        labels = F.one_hot(labels, 11).float()
        labels = Variable(labels, requires_grad=False).to(device)

        outputs = net2(images0)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        acc_batch = (outputs.cpu().argmax(-1) == labels.cpu().argmax(-1)).sum() / options_batch
        tr_acc_batches.append(acc_batch)

    correct = 0
    total = 0
    tr_acc_epoch = np.mean(tr_acc_batches) * 100
    tr_accs.append(tr_acc_epoch)
    tr_loss_epoch = running_loss / (len(train_dataset) - len(train_dataset) % options_batch)
    tr_losses.append(tr_loss_epoch)

    for images, labels in test_loader_crop:
        images0 = Variable(images.float(), requires_grad=True).to(device)
        labels = F.one_hot(labels, 11).float()
        labels = Variable(labels, requires_grad=False)

        outputs = net2(images0)

        te_loss = criterion(outputs.cpu(), labels)
        te_loss_epoch += te_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted.cpu() == labels).sum()

    te_acc = 100. * correct.float() / total

    te_accs.append(te_acc)
    te_loss_epoch = te_loss_epoch / len(test_dataset)
    te_losses.append(te_loss_epoch)

    save_str = ''
    if best_acc <= te_acc:
        best_acc = te_acc

        state = {
            'lsm_net': net.state_dict(),
            'readout_net': net2.state_dict(),
            'te_acc': te_acc,
            'epoch': epoch,
            'tr_accs': tr_accs,
            'tr_losses': tr_losses,
            'te_accs': te_accs,
            'te_losses': te_losses,
        }

        torch.save(state, f'{names}.t7')
        save_str = ', Model Saved.'
    if (epoch + 1) % 100 ==0:

        print(f'Epoch [{epoch + 1}/{options_epochs}], Train Loss: {tr_loss_epoch:.6f}, Train Acc: {tr_acc_epoch:.4f}, Test Loss: {te_loss_epoch:.6f}, Test Acc: {te_acc:.4f}, Best Test Acc: {best_acc:.4f}' + save_str)