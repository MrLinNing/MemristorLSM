import torch
import os
from models_suploss import CLIPModel, PrototypicalNetworks
import numpy as np
import random
from utils import AvgMeter
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import time
from datetime import datetime
from data_process_enlarge_22T4_sc import data_generate
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    ### Hyper-parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate ')
    parser.add_argument('--batch', type=int, default=1100,
                        help='batch size for input')
    parser.add_argument('--te_batch', type=int, default=600,
                        help='batch size for test')

    ### Parameters for  Readout layer
    parser.add_argument('--models', type=str, default='OneFC',
                        help='OneFC, TwoFC, ThreeFC')
    parser.add_argument('--pd', type=int, default=512,
                        help='projection dimention')
    parser.add_argument('--act', type=str, default='gelu',
                        help='relu, sigmoid, tanh, gelu')
    
    ### LOSS
    parser.add_argument('--temperature', type=float, default=0.001,
                        help='encoding feature batch size')
    
    ### Data process
    parser.add_argument('--enlarge_eeg_train', type=int, default=10,
                        help='Train Data Enlarge Factor for EEG')
    parser.add_argument('--enlarge_eeg_test', type=int, default=10,
                        help='Test Data Enlarge Factor for EEG')
    parser.add_argument('--eeg_std', type=float, default=0.0,
                        help='EEG data augmentation')
    parser.add_argument('--center_crop', type=int, default=28,
                        help='Center Crop for E-MNIST')
    
    ### cross train
    parser.add_argument('--in_class', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21',
                        help='in class dataset')
    parser.add_argument('--out_class', type=str, default='22,23,24,25',
                        help='out class dataset')
    
    ### eeg encoder
    parser.add_argument('--eeg_vth', type=float, default=0.1,
                        help='EEG embedding vth')
    parser.add_argument('--eeg_decay', type=float, default=0.95,
                        help='EEG embedding decay')
    parser.add_argument('--eeg_const', type=float, default=0.0005,
                        help='EEG embedding constant')
    parser.add_argument('--eeg_window', type=int, default=201,
                        help='time window for EEG')
    parser.add_argument('--eeg_ARCHI', type=str, default="192, 2048, 10", metavar='A',
                        help='Network node of eeg archi')
    
    
    ### vis encoder
    parser.add_argument('--vis_vth', type=float, default=0.1,
                        help='E-MNIST embedding vth')
    parser.add_argument('--vis_decay', type=float, default=0.95,
                        help='E-MNIST embedding decay')
    parser.add_argument('--vis_const', type=float, default=0.0005,
                        help='E-MNIST embedding constant')
    
    parser.add_argument('--max_spikes', type=int, default=20,
                        help='max spikes number for the max value in image')
    parser.add_argument('--emnist_window', type=int, default=50,
                        help='time window for E-MNIST')
    parser.add_argument('--emnist_ARCHI', type=str, default="784, 2048, 10", metavar='A',
                        help='Network node of emnist archi')


    ## configurations
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')

    parser.add_argument('--data_num', type=str, default="data_26", metavar='A',
                        help='data_10,data_26')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)

SAVE_DIR = f'CLIP_SRNN_checkpoint'

####set seed###
setup_seed(options.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')

SAVE_DIR = os.path.join(SAVE_DIR,
f'temperature{options.temperature}_eegconst{options.eeg_const}_emnistconst{options.vis_const}_pd{options.pd}')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')
device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")

### load dataset
train_loader_sup, test_loader_sup, train_loader_que, test_loader_que = data_generate(options)

### init clip model
N_ts = options.emnist_window  # Number of timesteps for spike trains
tw = N_ts-1

### EEG
eeg_cfg_net = [int(item) for item in options.eeg_ARCHI.split(',')]
print(f'net architecture for eeg is {eeg_cfg_net}')
### E-MNIST
emnist_cfg_net = [int(item) for item in options.emnist_ARCHI.split(',')]
print(f'net architecture for emnist is {emnist_cfg_net}')

net = PrototypicalNetworks(temper_factor=options.temperature, project_dim=options.pd, 
                eeg_cfg=eeg_cfg_net, vis_cfg=emnist_cfg_net,
                eeg_vth=options.eeg_vth, vis_vth=options.vis_vth,
                eeg_decay=options.eeg_decay, vis_decay=options.vis_decay,
                eeg_const=options.eeg_const, vis_const=options.vis_const,
                eeg_tw=options.eeg_window,  vis_tw=tw)
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
options_lr = options.lr
optimizer = torch.optim.Adam(net.parameters(), lr=options_lr)

options_epochs = options.epochs
options_batch = options.batch
print('batch size is',options_batch)


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    for i, (f_eegs, f_vis, eeg_labels, vis_labels) in enumerate(valid_loader):
        eegs0 = Variable(f_eegs, requires_grad=True).to(device)
        vis = Variable(f_vis, requires_grad=True).to(device)
        loss = model(eegs0, vis, eeg_labels)
        count = eegs0.size(0)
        loss_meter.update(loss.item(), count)
    return loss_meter


def valid_acc(model, img_que_v, aud_que_v, que_img_label, que_aud_label, top=[1, 5]):

    image_embeddings_n = F.normalize(img_que_v, p=2, dim=-1)
    audio_embeddings_n = F.normalize(aud_que_v, p=2, dim=-1)

    ### audio search image #########
    euc_dists = torch.cdist(audio_embeddings_n, image_embeddings_n, p=2)
    # top 5 accuracy
    indices_1 = torch.argmin(euc_dists, dim=1)
    indices_5 = torch.topk(euc_dists, k=top[1], dim=1, largest=False)[1]

    # get smallest k unique value in each row of input tensor
    # val, idx = torch.topk(euc_dists, k=top[1], dim=1, largest=False)

    # v5, indices_5 = torch.sorted(euc_dists, dim=1)
    # indices = torch.argmin(euc_dists, dim=1)
    img_label_1 = que_img_label[indices_1.cpu()]
    img_label_5 = que_img_label[indices_5.cpu()]
    correct_top1 = (img_label_1 == que_aud_label).sum().item()
    # correct_top5 = (que_aud_label == img_label_5).sum().item()
    correct_top5 = torch.where((img_label_5 == que_aud_label.unsqueeze(1)).sum(1))[0].shape[0]
    top_acc_a2i_top1 = correct_top1 / len(que_aud_label) * 100.0
    top_acc_a2i_top5 = correct_top5 / len(que_aud_label) * 100.0

    ### image search audio ########
    euc_dists = torch.cdist(image_embeddings_n, audio_embeddings_n, p=2)
    indices_1 = torch.argmin(euc_dists, dim=1)
    indices_5 = torch.topk(euc_dists, k=top[1], dim=1, largest=False)[1]
    aud_label_1 = que_aud_label[indices_1.cpu()]
    aud_label_5 = que_aud_label[indices_5.cpu()]
    correct_top1 = (aud_label_1 == que_img_label).sum().item()
    correct_top5 = torch.where((aud_label_5 == que_img_label.unsqueeze(1)).sum(1))[0].shape[0]
    top1_acc_i2a = correct_top1 / len(que_img_label) * 100.0
    top5_acc_i2a = correct_top5 / len(que_img_label) * 100.0

    return top_acc_a2i_top1, top_acc_a2i_top5, top1_acc_i2a, top5_acc_i2a

best_acc_top1_a2i = 0.0
best_acc_top1_i2a = 0.0

best_acc_top5_a2i = 0.0
best_acc_top5_i2a = 0.0

cur_out_acc_a2i = 0.0
cur_out_acc_i2a = 0.0
cur_out_acc_a2i_top5 = 0.0
cur_out_acc_i2a_top5 = 0.0



## save mode params
def save_model_parameters(model):
    return {name: param.clone() for name, param in model.named_parameters()}

## compute the diffsss
def print_parameters_diff(old_params, new_params):
    print("Parameters change:")
    for name in old_params:
        diff = torch.sum(torch.abs(new_params[name] - old_params[name])).item()
        print(f"{name}: {diff}")


# for epoch in range(options.epochs):
for epoch in range(options.epochs):
    print(f"Epoch: {epoch + 1}")

    net.train()

    old_params = save_model_parameters(net)
    correction = 0
    loss_meter = AvgMeter()
    tr_eeg_que_vs, tr_vis_que_vs, tr_eeg_que_ls, tr_vis_que_ls = [], [], [], []
    for i, ((sup_eeg, sup_vis, sup_eeg_labels, sup_vis_labels), (que_eeg, que_vis, que_eeg_labels, que_vis_labels)) in enumerate(zip(train_loader_sup, train_loader_que)):

        sup_eeg = Variable(sup_eeg, requires_grad=True).to(device)
        sup_vis = Variable(sup_vis, requires_grad=True).to(device)
        que_eeg = Variable(que_eeg, requires_grad=True).to(device)
        que_vis = Variable(que_vis, requires_grad=True).to(device)

        # tr_score, tr_proto, tr_img_que_v, tr_aud_que_v = net(f_eegs,f_vis, eeg_labels)
        tr_score, tr_proto, tr_eeg_que_v, tr_vis_que_v = net(sup_eeg, sup_vis, sup_eeg_labels, sup_vis_labels, que_eeg, que_vis)
        que_label = torch.cat([que_eeg_labels, que_vis_labels], dim=0).to(device, torch.int64)

        correction += (tr_score.argmax(dim=1) == que_label).sum().item()
        count = sup_eeg.size(0)
        loss = criterion(tr_score, que_label)
        loss_meter.update(loss.item(), count)
                # con_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tqdm_object.set_postfix(train_loss=loss_meter.avg)

        tr_eeg_que_vs.append(tr_eeg_que_v)
        tr_vis_que_vs.append(tr_vis_que_v)
        tr_eeg_que_ls.append(que_eeg_labels)
        tr_vis_que_ls.append(que_vis_labels)
    tr_eeg_que_vs, tr_vis_que_vs = torch.cat(tr_eeg_que_vs, dim=0), torch.cat(tr_vis_que_vs, dim=0)
    tr_eeg_que_ls, tr_vis_que_ls = torch.cat(tr_eeg_que_ls, dim=0), torch.cat(tr_vis_que_ls, dim=0)

    new_params = save_model_parameters(net)

    # print_parameters_diff(old_params, new_params)
    print(f"Train Loss is: {loss_meter.avg}, correction: {correction}, total: {len(train_loader_sup.dataset)}")

    net.eval()
    with torch.no_grad():
        # print("=============== LSM Encoding for test dataset =================")
        te_eeg_que_vs, te_vis_que_vs, te_eeg_que_ls, te_vis_que_ls = [], [], [], []
        for i, ((sup_eeg, sup_vis, sup_eeg_label, sup_vis_label), (que_eeg, que_vis, que_eeg_label, que_vis_label)) in enumerate(zip(test_loader_sup, test_loader_que)):

            sup_eeg = Variable(sup_eeg, requires_grad=False).to(device)
            sup_vis = Variable(sup_vis, requires_grad=False).to(device)
            que_eeg = Variable(que_eeg, requires_grad=False).to(device)
            que_vis = Variable(que_vis, requires_grad=False).to(device)

            te_score, te_proto, te_img_que_v, te_aud_que_v = net(sup_eeg, sup_vis, sup_eeg_label, sup_vis_label, que_eeg, que_vis, False)
            que_label = torch.cat([que_eeg_label, que_vis_label], dim=0).to(device, torch.int64)
            # correction
            correction += (te_score.argmax(dim=1) == que_label).sum().item()
            count = sup_eeg.size(0)

            te_eeg_que_vs.append(te_img_que_v)
            te_vis_que_vs.append(te_aud_que_v)
            te_eeg_que_ls.append(que_eeg_label)
            te_vis_que_ls.append(que_vis_label)

        te_eeg_que_vs, te_vis_que_vs = torch.cat(te_eeg_que_vs, dim=0), torch.cat(te_vis_que_vs, dim=0)
        te_eeg_que_ls, te_vis_que_ls = torch.cat(te_eeg_que_ls, dim=0), torch.cat(te_vis_que_ls, dim=0)


        # print(f"Test Loss is: {valid_loss.avg}")

        # in_acc_top1_a2i, in_acc_top1_i2a = valid_acc(net,
                                            #    tr_eeg_que_v, tr_vis_que_v, tr_que_,
                                            #    top=1)
        # print(f"Top1 Accuracy for vision search eeg (in-classes): {in_acc_top1_a2i}")
        # print(f"Top1 Accuracy for eeg search vision (in-classes): {in_acc_top1_i2a}")

        tr_que_acc_top1_a2i, tr_que_acc_top5_a2i, tr_que_acc_top1_i2a,  tr_que_acc_top5_i2a = valid_acc(net,
                                                 tr_eeg_que_vs, tr_vis_que_vs, tr_eeg_que_ls, tr_vis_que_ls,
                                               top=[1,5])
        te_que_acc_top1_a2i, te_que_acc_top5_a2i, te_que_acc_top1_i2a, te_que_acc_top5_i2a  = valid_acc(net,
                                                 te_eeg_que_vs, te_vis_que_vs, te_eeg_que_ls, te_vis_que_ls,
                                               top=[1,5])
        print(f"eeg search vision Top1 Accuracy in-classes: {tr_que_acc_top1_i2a}, top5: {tr_que_acc_top5_i2a}")
        print(f"eeg search vision Top1 Accuracy out-classes: {te_que_acc_top1_i2a}, top5: {te_que_acc_top5_i2a}")

        print(f"vision search eeg Top1 Accuracy in-classes: {tr_que_acc_top1_a2i}, top5: {tr_que_acc_top5_a2i}")
        print(f"vision search eeg Top1 Accuracy out-classes: {te_que_acc_top1_a2i}, top5: {te_que_acc_top5_a2i}")

    ### top1
    # if in_acc_top1_a2i >= 25 and out_acc_top1_a2i>=25:
    # if in_acc_top1_a2i >= 0 and out_acc_top1_a2i >= 0:
    if tr_que_acc_top1_a2i >= 0 and te_que_acc_top1_a2i >= 0:
        # if in_acc_top1_a2i > best_acc_top1_a2i:
        if tr_que_acc_top1_a2i > best_acc_top1_a2i:
            best_acc_top1_a2i = tr_que_acc_top1_a2i # in_acc_top1_a2i
            cur_out_acc_a2i = te_que_acc_top1_a2i # out_acc_top1_a2i

            best_acc_top5_a2i = tr_que_acc_top5_a2i # in_acc_top1_a2i
            cur_out_acc_a2i_top5 = te_que_acc_top5_a2i # out_acc_top1_a2i


            state = {
                'net': net.state_dict(),
                'top1_in_acc': best_acc_top1_a2i,
                'top1_out_acc': cur_out_acc_a2i,
                'top5_in_acc': best_acc_top5_a2i,
                'top5_out_acc': cur_out_acc_a2i_top5,
            }
            torch.save(state, f'{names}_top1_a2i.t7')
            print("Saved Best Model!")
            print('In-classes vision search eeg best accuracy is', best_acc_top1_a2i)
            print('Out-classes vision search eeg best accuracy is', cur_out_acc_a2i)
            print('In-cllasses vision search eeg top5 accuracy is', tr_que_acc_top5_a2i)
            print('out-cllasses vision search eeg top5 accuracy is', te_que_acc_top5_a2i)

    ### top1
    # if in_acc_top1_i2a >= 0 and out_acc_top1_i2a >=0:
    if tr_que_acc_top1_i2a >= 0 and te_que_acc_top1_i2a >=0:
    # if in_acc_top1_i2a >= 0 and out_acc_top1_i2a >= 0:
        if tr_que_acc_top1_i2a > best_acc_top1_i2a:
            best_acc_top1_i2a = tr_que_acc_top1_i2a 
            cur_out_acc_i2a = te_que_acc_top1_i2a

            best_acc_top5_i2a = tr_que_acc_top5_i2a
            cur_out_acc_i2a_top5 = te_que_acc_top5_i2a

            state = {
                'net': net.state_dict(),
                'top1_in_acc': best_acc_top1_i2a,
                'top1_out_acc': cur_out_acc_i2a,
                'top5_in_acc': best_acc_top5_i2a,
                'top5_out_acc': cur_out_acc_i2a_top5,

            }
            torch.save(state, f'{names}_top1_i2a.t7')
            print("Saved Best Model!")
            print('In-classes eeg search vision best accuracy is', best_acc_top1_i2a)
            print('Out-classes eeg search vision best accuracy is', cur_out_acc_i2a)
            print('In-cllasses eeg - v top5 accuracy is', tr_que_acc_top5_i2a)
            print('out-cllasses eeg - vis top5 accuracy is', te_que_acc_top5_i2a)

print(f"Finally In Top1-acc for E-MNIST search EEG: {best_acc_top1_a2i}")
print(f"Finally In Top1-acc for EEG search E-MNIST: {best_acc_top1_i2a}")
print(f"Finally Out Top1-acc for E-MNIST search EEG: {cur_out_acc_a2i}")
print(f"Finally Out Top1-acc for EEG search E-MNIST: {cur_out_acc_i2a}")

print(f"Finally In Top5-acc for E-MNIST search EEG: {best_acc_top5_a2i}")
print(f"Finally In Top5-acc for EEG search E-MNIST: {best_acc_top5_i2a}")
print(f"Finally Out Top5-acc for E-MNIST search EEG: {cur_out_acc_a2i_top5}")
print(f"Finally Out Top5-acc for EEG search E-MNIST: {cur_out_acc_i2a_top5}")
