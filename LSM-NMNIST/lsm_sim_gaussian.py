import torch
import time
from datetime import datetime
import os
import argparse
import torch.nn as nn
from torch.autograd import Variable
from utils import CustomTensorDataset
import random
import numpy as np

from models import simulation_Gua_LSM_Model,Read_ANN
from utils import MyCenterCropDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    ## hyper-parameters
    parser.add_argument('--batch', type=int, default=1000,
                        help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate ')
    parser.add_argument('--ann_batch', type=int, default=100,
                        help='input batch size for training default (128)')

    ## Model parameters
    parser.add_argument('--scale', type=float, default=100.0,
                        help='scale factor (default: 100.0)')
    parser.add_argument('--Decay_lsm', type=float, default=0.9,
                        help='Decay Value of LIF for LSM')
    parser.add_argument('--Vth', type=float, default=5,
                        help='Vth value of LIF')
    parser.add_argument('--Tw', type=int, default=50,
                        help='Time window value of LIF')
    parser.add_argument('--const', type=float, default=1.0,
                        help='Vth value of LIF')
    parser.add_argument('--ARCHI', type=str, default="256,200,10",
                        help='Network node of SNN')
    parser.add_argument('--input_size', type=int, default=16,
                        help='center crop for input')


    ## training configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='seed id')


    args = parser.parse_args()
    return args


options = parse_args()
print(options)

os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
setup_seed(options.seed)


SAVE_DIR = 'Gau_LSM_Sim_Checkpoint'

cfg_net = [int(item) for item in options.ARCHI.split(',')]
print(cfg_net)


t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR,
                        f'seed{options.seed}_input{options.input_size}_b{options.ann_batch}_timewindow{options.Tw}_decaylsm{options.Decay_lsm}_lr{options.lr}_ARCHI{cfg_net}_Vth{options.Vth}_const{options.const}_scale{options.scale}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')

device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")
train_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_train_data.mat'
test_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_test_data.mat'


train_dataset_crop = MyCenterCropDataset(
    train_path, 'nmnist_h', new_width=options.input_size, new_height=options.input_size)
# test data size: 10k
test_dataset_crop = MyCenterCropDataset(
    test_path, 'nmnist_r', new_width=options.input_size, new_height=options.input_size)

train_loader_crop = torch.utils.data.DataLoader(dataset=train_dataset_crop,
                                                batch_size=options.batch,
                                                shuffle=True, drop_last=True)
test_loader_crop = torch.utils.data.DataLoader(dataset=test_dataset_crop,
                                               batch_size=options.batch,
                                               shuffle=False, drop_last=False)

net = simulation_Gua_LSM_Model(cfg=cfg_net, time_window=options.Tw, lens=0.25, thresh=options.Vth,
                               decay_lsm=options.Decay_lsm, const=options.const, scale=options.scale)
net.to(device)

print('==============================Begin process by the LSM========================================')
################# process by the RRAM-LSM ###################

### train data###
counter_train_data_list = []
counter_train_label_list = []
for i, (images, labels) in enumerate(train_loader_crop):
    net.zero_grad()
    images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)

    outputs = net(images0)
    # print(outputs.shape)
    counter_train_data_list.append(outputs)
    counter_train_label_list.append(labels)

counter_train_data_list_tensor = torch.cat(counter_train_data_list,0)
print(counter_train_data_list_tensor.shape)

counter_train_label_list_tensor = torch.cat(counter_train_label_list,0)
print(counter_train_label_list_tensor.shape)


### test data ####
counter_val_data_list = []
counter_val_label_list = []

for i, (images, labels) in enumerate(test_loader_crop):
    net.zero_grad()
    images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=False).to(device)

    outputs = net(images0)
    # print(outputs.shape)
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
                                          batch_size=options_batch
                                          )
test_loader_crop = torch.utils.data.DataLoader(test_dataset_normal,
                                                batch_size=options_batch
                                               )


net2 = Read_ANN(cfg=cfg_net)
net2.to(device)

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=options.lr)


options_lr = options.lr

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net2.parameters(), lr=options_lr)

options_epochs = options.epochs


best_acc = 0
tr_accs, te_accs, tr_losses, te_losses = [], [], [], []
for epoch in range(options_epochs):
    running_loss, te_loss_epoch = 0, 0
    start_time = time.time()
    tr_acc_batches = []
    for i, (images, labels) in enumerate(train_loader_crop):
        net2.zero_grad()
        optimizer.zero_grad()
        images0 = Variable(images.float(), requires_grad=True).to(device)
        # print('label is',labels)
        # labels = F.one_hot(labels, 10).float()
        labels = Variable(labels, requires_grad=False).to(device)

        outputs = net2(images0)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # labels = F.one_hot(labels, 11).float()
        acc_batch = (outputs.cpu().argmax(-1) == labels.cpu().argmax(-1)).sum() / options_batch
        tr_acc_batches.append(acc_batch)

    correct = 0
    total = 0
    tr_acc_epoch = np.mean(tr_acc_batches) * 100
    tr_accs.append(tr_acc_epoch)
    tr_loss_epoch = running_loss / (len(train_dataset_crop) - len(train_dataset_crop) % options_batch)
    tr_losses.append(tr_loss_epoch)

    # optimizer = lr_scheduler(optimizer, epoch, options.lr, 50)
    for images, labels in test_loader_crop:
        images0 = Variable(images.float(), requires_grad=True).to(device)
        # print('label is', labels)
        # labels = F.one_hot(labels, 10).float()
        labels = Variable(labels, requires_grad=False)

        outputs = net2(images0)

        te_loss = criterion(outputs.cpu(), labels)
        te_loss_epoch += te_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        # labels = F.one_hot(labels, 11).float()
        _, labels = torch.max(labels.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted.cpu() == labels).sum()

    te_acc = 100. * correct.float() / total

    te_accs.append(te_acc)
    te_loss_epoch = te_loss_epoch / len(test_dataset_crop)
    te_losses.append(te_loss_epoch)

    save_str = ''
    if best_acc <= te_acc:
        best_acc = te_acc

        state = {
            'lsm_net': net.state_dict(),
            'readout_net': net2.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch,
            'tr_accs': tr_accs,
            'tr_losses': tr_losses,
            'te_accs': te_accs,
            'te_losses': te_losses,
        }

        torch.save(state, f'{names}.t7')
        save_str = ', Model Saved.'
    if (epoch + 1) % 50 ==0:

        print(
        f'Epoch [{epoch + 1}/{options_epochs}], Train Loss: {tr_loss_epoch:.6f}, Train Acc: {tr_acc_epoch:.4f}, Test Loss: {te_loss_epoch:.6f}, Test Acc: {te_acc:.4f}, Best Test Acc: {best_acc:.4f}' + save_str)