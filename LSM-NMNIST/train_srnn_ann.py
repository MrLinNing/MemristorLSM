import torch
import time
from datetime import datetime
import os
import argparse
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
### self-define libs
from models import SRNN_ANN_Model
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
    parser.add_argument('--batch', type=int, default=80,
                        help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate ')
    ## SNN parameters
    parser.add_argument('--scale', type=float, default=100.0,
                        help='scale factor (default: 100.0)')
    parser.add_argument('--Decay_lsm', type=float, default=0.9,
                        help='Decay Value of LIF for LSM')
    parser.add_argument('--Vth', type=float, default=0.3,
                        help='Vth value of LIF')
    parser.add_argument('--Tw', type=int, default=50,
                        help='Time window value of LIF')
    parser.add_argument('--const', type=float, default=1.0,
                        help='Vth value of LIF')
    parser.add_argument('--ARCHI', type=str, default="256,200,10",
                        help='Network node of SNN')
    ## training configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='GPU id')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
setup_seed(options.seed)

SAVE_DIR = 'SRNN_ANN_Checkpoint'
cfg_net = [int(item) for item in options.ARCHI.split(',')]
print(cfg_net)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR,
                        f'b{options.batch}_timewindow{options.Tw}_decaylsm{options.Decay_lsm}_lr{options.lr}_ARCHI{cfg_net}_Vth{options.Vth}_const{options.const}_scale{options.scale}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')

device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")
train_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_train_data.mat'
test_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_test_data.mat'

train_dataset_crop = MyCenterCropDataset(
    train_path, 'nmnist_h', new_width=16, new_height=16)
# test data size: 10k
test_dataset_crop = MyCenterCropDataset(
    test_path, 'nmnist_r', new_width=16, new_height=16)

train_loader_crop = torch.utils.data.DataLoader(dataset=train_dataset_crop,
                                                batch_size=options.batch,
                                                shuffle=True, drop_last=True)
test_loader_crop = torch.utils.data.DataLoader(dataset=test_dataset_crop,
                                               batch_size=options.batch,
                                               shuffle=False, drop_last=False)

net = SRNN_ANN_Model(cfg=cfg_net, time_window=options.Tw, lens=0.25, thresh=options.Vth,
                     decay_lsm=options.Decay_lsm, const=options.const, scale=options.scale)
net.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=options.lr)

best_acc = 0
tr_accs, te_accs, tr_losses, te_losses = [], [], [], []
for epoch in range(options.epochs):
    running_loss, te_loss_epoch = 0, 0
    start_time = time.time()
    tr_acc_batches = []
    for i, (images, labels) in enumerate(train_loader_crop):
        net.zero_grad()
        optimizer.zero_grad()

        images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)
        labels = Variable(labels, requires_grad=True).to(device)

        outputs = net(images0)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        acc_batch = (outputs.cpu().argmax(-1) == labels.cpu().argmax(-1)).sum() / options.batch
        tr_acc_batches.append(acc_batch)

    correct = 0
    total = 0
    tr_acc_epoch = np.mean(tr_acc_batches) * 100
    tr_accs.append(tr_acc_epoch)
    tr_loss_epoch = running_loss / (len(train_dataset_crop) - len(train_dataset_crop) % options.batch)
    tr_losses.append(tr_loss_epoch)

    for images, labels in test_loader_crop:
        images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(),requires_grad=False).to(device)
        outputs = net(images0)
        te_loss = criterion(outputs.cpu(), labels)
        te_loss_epoch += te_loss.item()
        _, predicted = torch.max(outputs.data, 1)
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
            'net': net.state_dict(),
            'te_acc': te_acc,
            'epoch': epoch,
            'tr_accs': tr_accs,
            'tr_losses': tr_losses,
            'te_accs': te_accs,
            'te_losses': te_losses,
        }

        torch.save(state, f'{names}.t7')
        save_str = ', Model Saved.'
    print(f'Epoch [{epoch + 1}/{options.epochs}], Train Loss: {tr_loss_epoch:.6f}, Train Acc: {tr_acc_epoch:.4f}, Test Loss: {te_loss_epoch:.6f}, Test Acc: {te_acc:.4f}, Best Test Acc: {best_acc:.4f}' + save_str)




