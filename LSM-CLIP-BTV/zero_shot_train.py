import torch
import os
from models_suploss import CLIPModel
import numpy as np
import random
from utils import AvgMeter
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import time
from datetime import datetime
from data_process_enlarge_22T4 import data_generate
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
    parser.add_argument('--batch', type=int, default=24,
                        help='batch size for input')

    ### Parameters for  Readout layer
    parser.add_argument('--models', type=str, default='OneFC',
                        help='OneFC, TwoFC, ThreeFC')
    parser.add_argument('--pd', type=int, default=64,
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
train_loader_crop, test_loader_crop_in, test_loader_crop_out = data_generate(options)

### init clip model
N_ts = options.emnist_window  # Number of timesteps for spike trains
tw = N_ts-1

### EEG
eeg_cfg_net = [int(item) for item in options.eeg_ARCHI.split(',')]
print(f'net architecture for eeg is {eeg_cfg_net}')
### E-MNIST
emnist_cfg_net = [int(item) for item in options.emnist_ARCHI.split(',')]
print(f'net architecture for emnist is {emnist_cfg_net}')

net = CLIPModel(temper_factor=options.temperature, project_dim=options.pd, 
                eeg_cfg=eeg_cfg_net, vis_cfg=emnist_cfg_net,
                eeg_vth=options.eeg_vth, vis_vth=options.vis_vth,
                eeg_decay=options.eeg_decay, vis_decay=options.vis_decay,
                eeg_const=options.eeg_const, vis_const=options.vis_const,
                eeg_tw=options.eeg_window,  vis_tw=tw
                )
net.to(device)


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




def valid_acc(model,
              valid_loader,
              top=1):
    
    counter_eeg_test_data_list = []
    counter_eeg_test_label_list = []

    counter_vis_test_data_list = []
    counter_vis_test_label_list = []


    for i, (eegs, vis, eeg_labels, vis_labels) in enumerate(valid_loader):

        f_eegs = Variable(eegs, requires_grad=True).to(device)
        f_vis = Variable(vis, requires_grad=True).to(device)

        eeg_ebd = model.eeg_encoder(f_eegs)
        vis_ebd = model.vis_encoder(f_vis)

        eeg_embeddings = model.eeg_projection(eeg_ebd)
        vis_embeddings = model.vis_projection(vis_ebd)


        counter_eeg_test_data_list.append(eeg_embeddings)
        counter_vis_test_data_list.append(vis_embeddings)
        counter_eeg_test_label_list.append(eeg_labels)
        counter_vis_test_label_list.append(vis_labels)


    counter_eeg_test_data_list_tensor = torch.cat(counter_eeg_test_data_list, 0)
    # print(counter_eeg_test_data_list_tensor.shape)
    counter_eeg_test_label_list_tensor = torch.cat(counter_eeg_test_label_list, 0)
    # print(counter_eeg_test_label_list_tensor.shape)

    counter_vis_test_data_list_tensor = torch.cat(counter_vis_test_data_list, 0)
    # print(counter_vis_test_data_list_tensor.shape)
    counter_vis_test_label_list_tensor = torch.cat(counter_vis_test_label_list, 0)


    # normalized features
    eeg_embeddings_n = counter_eeg_test_data_list_tensor / counter_eeg_test_data_list_tensor.norm(dim=-1, keepdim=True)
    vis_embeddings_n = counter_vis_test_data_list_tensor / counter_vis_test_data_list_tensor.norm(dim=-1, keepdim=True)


    # eeg_embeddings_n = F.normalize(eeg_embeddings, p=2, dim=-1)
    # vis_embeddings_n = F.normalize(vis_embeddings, p=2, dim=-1)

    ### vision search eeg #########
    count_m = 0
    logit_scale = model.logit_scale.exp()
    for idx in range(len(counter_vis_test_label_list_tensor)):

        
        eeg_label = counter_eeg_test_label_list_tensor[idx]

        # print(f"image label is {img_label}")

        dot_similarity = vis_embeddings_n[idx] @ eeg_embeddings_n.T

        # print(f"dot_similarity is {dot_similarity}")

        values, indices = torch.topk(logit_scale*dot_similarity.squeeze(0), top)
        # print(f"indices is {indices}")

        pre = counter_eeg_test_label_list_tensor[indices.cpu()]

        # print(f"prediction label is {pre}")
        # print(f"true label is {img_label}")


        if eeg_label.cpu() in pre:
            count_m = count_m + 1
        # else:
        #     # print("no")
    top_acc_i2b = count_m/len(counter_vis_test_label_list_tensor)*100.0

    # print(f"len(counter_vis_test_label_list_tensor) is {len(counter_vis_test_label_list_tensor)}, count_m is {count_m}")

    ### eeg search vis ########
    count_aud = 0
    for idx in range(len(counter_eeg_test_label_list_tensor)):
        vis_label = counter_vis_test_label_list_tensor[idx]

        dot_similarity = eeg_embeddings_n[idx] @ vis_embeddings_n.T

        values, indices = torch.topk(logit_scale*dot_similarity.squeeze(0), top)
        pre = counter_vis_test_label_list_tensor[indices.cpu()]

        if vis_label.cpu() in pre:
            # print("yes")
            count_aud = count_aud + 1
        # else:
        #     # print("no")
    top_acc_b2i = count_aud / len(counter_eeg_test_label_list_tensor) * 100.0

    return top_acc_i2b, top_acc_b2i



best_acc_top1_v2e = 0.0
best_acc_top1_e2v = 0.0

best_acc_top5_b2i = 0.0
best_acc_top5_e2v = 0.0

cur_out_acc_v2e = 0.0
cur_out_acc_e2v = 0.0

cur_out_acc_v2e_top5 = 0.0
cur_out_acc_e2v_top5 = 0.0


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

    loss_meter = AvgMeter()
    for i, (f_eegs, f_vis, eeg_labels, vis_labels) in enumerate(train_loader_crop):

        f_eegs = Variable(f_eegs, requires_grad=True).to(device)
        f_vis = Variable(f_vis, requires_grad=True).to(device)

        # print(f"image shape is {f_eegs.shape}")
        # print(f"image value is {f_eegs}")
        # print(f"audio shape is {f_vis.shape}")
        # print(f"audio value is {f_vis}")

        loss = net(f_eegs,f_vis, eeg_labels)
        # print(f"train loss is {loss.item()}")

        count = f_eegs.size(0)
        loss_meter.update(loss.item(), count)
                # con_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tqdm_object.set_postfix(train_loss=loss_meter.avg)

    new_params = save_model_parameters(net)
    
    print_parameters_diff(old_params, new_params)
    
    print(f"Train Loss is: {loss_meter.avg}")


    net.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(net, test_loader_crop_in)
        print(f"Test Loss is: {valid_loss.avg}")

        in_acc_top1_v2e, in_acc_top1_e2v = valid_acc(net,
                                               test_loader_crop_in,
                                               top=1)
        in_acc_top1_v2e_t5, in_acc_top1_e2v_t5 = valid_acc(net,
                                               test_loader_crop_in,
                                               top=5)

        print(f"Top1 Accuracy for vision search eeg (in-classes): {in_acc_top1_v2e}")
        print(f"Top1 Accuracy for eeg search vision (in-classes): {in_acc_top1_e2v}")

        print(f"Top5 Accuracy for vision search eeg (in-classes): {in_acc_top1_v2e_t5}")
        print(f"Top5 Accuracy for eeg search vision (in-classes): {in_acc_top1_e2v_t5}")


        out_acc_top1_v2e, out_acc_top1_e2v = valid_acc(net,
                                               test_loader_crop_out,
                                               top=1)
        out_acc_top1_v2e_t5, out_acc_top1_e2v_t5 = valid_acc(net,
                                               test_loader_crop_out,
                                               top=5)
        print(f"Top1 Accuracy for vision search eeg (out-classes): {out_acc_top1_v2e}")
        print(f"Top1 Accuracy for eeg search vision (out-classes): {out_acc_top1_e2v}")

        print(f"Top5 Accuracy for vision search eeg (out-classes): {out_acc_top1_v2e_t5}")
        print(f"Top5 Accuracy for eeg search vision (out-classes): {out_acc_top1_e2v_t5}")


        ### top1
    # if in_acc_top1_v2e >= 25 and out_acc_top1_v2e>=25:
    if in_acc_top1_v2e >= 0 and out_acc_top1_v2e >= 0:
        if in_acc_top1_v2e > best_acc_top1_v2e:
            best_acc_top1_v2e = in_acc_top1_v2e
            cur_out_acc_v2e = out_acc_top1_v2e

            best_acc_top5_b2i = in_acc_top1_v2e_t5
            cur_out_acc_v2e_top5 = out_acc_top1_v2e_t5

            state = {
                'net': net.state_dict(),
                'in_acc': best_acc_top1_v2e,
                'in_acc_top5': best_acc_top5_b2i,
                'out_acc_top1': cur_out_acc_v2e,
                'out_acc_top5': cur_out_acc_v2e_top5
            }
            torch.save(state, f'{names}_top1_a2i.t7')
            print("Saved Best Model!")

            print('In-classes vision search eeg best accuracy is', best_acc_top1_v2e)
            print('Out-classes vision search eeg best accuracy is', cur_out_acc_v2e)


    ### top1
    if in_acc_top1_e2v >= 0 and out_acc_top1_e2v >=0:
    # if in_acc_top1_e2v >= 0 and out_acc_top1_e2v >= 0:
        if in_acc_top1_e2v > best_acc_top1_e2v:
            best_acc_top1_e2v = in_acc_top1_e2v
            cur_out_acc_e2v = out_acc_top1_e2v

            best_acc_top5_e2v = in_acc_top1_e2v_t5
            cur_out_acc_e2v_top5 = out_acc_top1_e2v_t5

            state = {
                'net': net.state_dict(),
                'in_acc': best_acc_top1_e2v,
                'in_acc_top5': best_acc_top5_e2v,
                'out_acc_top1': cur_out_acc_e2v,
                'out_acc_top5': cur_out_acc_e2v_top5
            }
            torch.save(state, f'{names}_top1_i2a.t7')
            print("Saved Best Model!")

            print('In-classes eeg search vision best accuracy is', best_acc_top1_e2v)
            print('Out-classes eeg search vision best accuracy is', cur_out_acc_e2v)

print(f"Finally In Top1-acc for E-MNIST search EEG: {best_acc_top1_v2e}")
print(f"Finally In Top1-acc for EEG search E-MNIST: {best_acc_top1_e2v}")
print(f"Finally Out Top1-acc for E-MNIST search EEG: {cur_out_acc_v2e}")
print(f"Finally Out Top1-acc for EEG search E-MNIST: {cur_out_acc_e2v}")


print(f"TOP5 Finally In Top1-acc for E-MNIST search EEG: {best_acc_top5_b2i}")
print(f"TOP5 Finally In Top1-acc for EEG search E-MNIST: {best_acc_top5_e2v}")
print(f"TOP5 Finally Out Top1-acc for E-MNIST search EEG: {cur_out_acc_v2e_top5}")
print(f"TOP5 Finally Out Top1-acc for EEG search E-MNIST: {cur_out_acc_e2v_top5}")



