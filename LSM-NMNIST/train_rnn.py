import torch
import argparse
import torch.nn as nn
import time
import os
from datetime import datetime
from torch.autograd import Variable
import random
import numpy as np

from utils import MyCenterCropDataset
from models import RNN_Model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def arg_parse():
    parser = argparse.ArgumentParser()

    ## hyper-parameters
    parser.add_argument('--batch', type=int, default=80,
                        help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate ')
    parser.add_argument('--Tw', type=int, default=50,
                        help='Time window value of LIF')
    ## model information
    parser.add_argument('--ARCHI', type=str, default="256,200,10",
                        help='Network node')
    ## training configurations
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id')
    parser.add_argument('--seed', type=int, default=10, help='GPU id')

    args = parser.parse_args()
    return args

options = arg_parse()


os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)
setup_seed(options.seed)

batch_size = options.batch
num_epochs = options.epochs
learning_rate = options.lr
time_window = options.Tw
cfg_net = [int(item) for item in options.ARCHI.split(',')]
print(cfg_net)



SAVE_DIR = 'RNN_Checkpoint'
t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')
SAVE_DIR = os.path.join(SAVE_DIR, f'b{options.batch}_timewindow{options.Tw}_lr{options.lr}_ARCHI{cfg_net}_{t}')
os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_train_data.mat'
test_path = '/home/rram/Nature-LSM/NMNIST/data/NMNIST_test_data.mat'

train_dataset_crop = MyCenterCropDataset(
    train_path, 'nmnist_h', new_width=16, new_height=16)
# test data size: 10k
test_dataset_crop = MyCenterCropDataset(
    test_path, 'nmnist_r', new_width=16, new_height=16)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset_crop,
                                          batch_size=batch_size,
                                          shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset_crop,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)


best_acc = 0
acc_record = list([])

rnn_model = RNN_Model(cfg=cfg_net)
rnn_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

best_acc = 0
for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        rnn_model.zero_grad()
        optimizer.zero_grad()


        images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)

        outputs, _ = rnn_model(images0,time_window)
        loss = criterion(outputs.cpu(), labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch+1, num_epochs, i+1, len(train_dataset_crop)//batch_size, running_loss))
            running_loss = 0
            print('Time elasped:', time.time()-start_time)


    correct = 0
    total = 0

    for images, labels in test_loader:
        images0 = Variable(images[:, 0, :, :, :].unsqueeze(1).float(), requires_grad=True).to(device)

        outputs, _ = rnn_model(
            images0, time_window)  # todo  # todo
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total += float(labels.size(0))
        correct += (predicted.cpu() == labels).sum()
    print('Iters:', epoch, '\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' %
          (100 * correct.float() / total))

    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

    if best_acc <= acc:
        best_acc = acc

        print('Best Acc is',acc)
        print('Saving..')
        state = {
            'net': rnn_model.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }

        torch.save(state, f'{names}.t7')
        best_acc = acc