import torch
import os
from models_suploss import LSM_Vision_Encoder,LSM_EEG_Encoder
from models_suploss import Contrastive_Simple,Contrastive_twolayers,Contrastive_threelayers
import numpy as np
import random
from utils import CustomTensorDataset_clip
from utils import AvgMeter
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import time
from datetime import datetime
# from data_process_enlarge_all import data_generate
# from data_process_enlarge_24T2 import data_generate
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
    parser.add_argument('--batch', type=int, default=512,
                        help='batch size for input')

    ### Parameters for LSM
    parser.add_argument('--eeg_const', type=float, default=0.0005,
                        help='image encoder weight constant')
    parser.add_argument('--emnist_const', type=float, default=0.0005,
                        help='vision encoder weight constant')
    # decay
    parser.add_argument('--eeg_decay', type=float, default=0.95,
                        help='image encoder decay')
    parser.add_argument('--emnist_decay', type=float, default=0.95,
                        help='vision encoder decay')
    # vth
    parser.add_argument('--eeg_vth', type=float, default=0.1,
                        help='image vth')
    parser.add_argument('--emnist_vth', type=float, default=0.1,
                        help='image vth')
    # time window
    parser.add_argument('--eeg_window', type=int, default=201,
                        help='time window for EEG')
    
    parser.add_argument('--max_spikes', type=int, default=20,
                        help='max spikes number for the max value in image')
    parser.add_argument('--emnist_window', type=int, default=50,
                        help='time window for E-MNIST')
    
    # archi
    parser.add_argument('--eeg_ARCHI', type=str, default="192, 2048, 10", metavar='A',
                        help='Network node of eeg archi')
    parser.add_argument('--emnist_ARCHI', type=str, default="784, 2048, 10", metavar='A',
                        help='Network node of emnist archi')


    ### Parameters for  Readout layer
    parser.add_argument('--models', type=str, default='OneFC',
                        help='OneFC, TwoFC, ThreeFC')
    parser.add_argument('--pd', type=int, default=64,
                        help='projection dimention')
    parser.add_argument('--act', type=str, default='gelu',
                        help='relu, sigmoid, tanh, gelu')
    parser.add_argument('--feature_batch', type=int, default=24,
                        help='encoding feature batch size')
    
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



SAVE_DIR = f'sup_zero_shot_{options.data_num}_EEGstd{options.eeg_std}_simulation_checkpoint'


####set seed###
setup_seed(options.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(options.gpu_id)

t = datetime.fromtimestamp(time.time())
t = datetime.strftime(t, '%Y%m%d%H%MS')


SAVE_DIR = os.path.join(SAVE_DIR,
f'temperature{options.temperature}_eegconst{options.eeg_const}_emnistconst{options.emnist_const}_model{options.models}_pd{options.pd}')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

names = os.path.join(SAVE_DIR, 'checkpoint')
device = torch.device(options.cuda if torch.cuda.is_available() else "cpu")




train_loader_crop, test_loader_crop_in, test_loader_crop_out = data_generate(options)

### EEG
eeg_cfg_net = [int(item) for item in options.eeg_ARCHI.split(',')]
print(f'net architecture for eeg is {eeg_cfg_net}')

eeg_model = LSM_EEG_Encoder(cfg=eeg_cfg_net,time_window=options.eeg_window, thresh=options.eeg_vth,
                  decay_lsm =options.eeg_decay, const = options.eeg_const)

### E-MNIST
emnist_cfg_net = [int(item) for item in options.emnist_ARCHI.split(',')]
print(f'net architecture for emnist is {emnist_cfg_net}')

N_ts = options.emnist_window  # Number of timesteps for spike trains
tw = N_ts-1
vision_model = LSM_Vision_Encoder(cfg=emnist_cfg_net,time_window=tw, thresh=options.emnist_vth,
                                  decay_lsm =options.emnist_decay, const = options.emnist_const)




###################### LSM encoder #############################
print("=============== LSM Encoding for train dataset =================")

counter_eeg_train_data_list = []
counter_eeg_train_label_list = []

counter_vis_train_data_list = []
counter_vis_train_label_list = []

eeg_model.eval()
vision_model.eval()
for i, (eeg, vision, eeg_labels, vision_labels) in enumerate(train_loader_crop):

    # print(f"EEG data is {eeg}")
    print(f"EEG label is {eeg_labels}")

    # print(f"Vision data is {vision}")
    print(f"Vision data label is {vision_labels}")

    eeg = Variable(eeg.float(), requires_grad=False)
    vision = Variable(vision.float(), requires_grad=False)

    eeg_fea = eeg_model(eeg)
    vis_fea = vision_model(vision)

    counter_eeg_train_data_list.append(eeg_fea)
    counter_eeg_train_label_list.append(eeg_labels)

    counter_vis_train_data_list.append(vis_fea)
    counter_vis_train_label_list.append(vision_labels)


counter_eeg_train_data_list_tensor = torch.cat(counter_eeg_train_data_list,0)
print(f"EEG train data shape is {counter_eeg_train_data_list_tensor.shape}")
counter_eeg_train_label_list_tensor = torch.cat(counter_eeg_train_label_list,0)
print(f"EEG train label shape is {counter_eeg_train_label_list_tensor.shape}")

counter_vis_train_data_list_tensor = torch.cat(counter_vis_train_data_list,0)
print(f"E-MNIST train data shape is {counter_vis_train_data_list_tensor.shape}")
counter_vis_train_label_list_tensor = torch.cat(counter_vis_train_label_list,0)
print(f"E-MNIST train label is {counter_vis_train_label_list_tensor.shape}")

print("=============== LSM Encoding for test dataset (in classes) =================")

in_counter_eeg_test_data_list = []
in_counter_eeg_test_label_list = []

in_counter_vis_test_data_list = []
in_counter_vis_test_label_list = []

eeg_model.eval()
vision_model.eval()
for i, (eeg, vision, eeg_labels, vision_labels) in enumerate(test_loader_crop_in):
    # print(f"EEG data is {eeg}")
    # print(f"EEG label is {eeg_labels}")
    # print(f"Vision data is {vision}")
    # print(f"Vision data label is {vision_labels}")

    eeg = Variable(eeg.float(), requires_grad=False)
    vision = Variable(vision.float(), requires_grad=False)

    eeg_fea = eeg_model(eeg)
    vis_fea = vision_model(vision)

    in_counter_eeg_test_data_list.append(eeg_fea)
    in_counter_eeg_test_label_list.append(eeg_labels)

    in_counter_vis_test_data_list.append(vis_fea)
    in_counter_vis_test_label_list.append(vision_labels)


in_counter_eeg_test_data_list_tensor = torch.cat(in_counter_eeg_test_data_list,0)
print(f"EEG in-class test data shape is {in_counter_eeg_test_data_list_tensor.shape}")
in_counter_eeg_test_label_list_tensor = torch.cat(in_counter_eeg_test_label_list,0)
print(f"EEG in-class test label shape is {in_counter_eeg_test_label_list_tensor.shape}")

in_counter_vis_test_data_list_tensor = torch.cat(in_counter_vis_test_data_list,0)
print(f"E-MNIST in-class test data shape is {in_counter_vis_test_data_list_tensor.shape}")
in_counter_vis_test_label_list_tensor = torch.cat(in_counter_vis_test_label_list,0)
print(f"E-MNIST in-class test label shape is {in_counter_vis_test_label_list_tensor.shape}")



print("=============== LSM Encoding for test dataset (out classes) =================")

out_counter_eeg_test_data_list = []
out_counter_eeg_test_label_list = []

out_counter_vis_test_data_list = []
out_counter_vis_test_label_list = []

eeg_model = eeg_model.to(device)
vision_model = vision_model.to(device)

for i, (eeg, vision, eeg_labels, vision_labels) in enumerate(test_loader_crop_out):

    # print(f"EEG data is {eeg}")
    # print(f"EEG label is {eeg_labels}")
    # print(f"Vision data is {vision}")
    # print(f"Vision data label is {vision_labels}")

    eeg = Variable(eeg.float(), requires_grad=True).to(device)
    vision = Variable(vision.float(), requires_grad=True).to(device)

    eeg_fea = eeg_model(eeg)
    vis_fea = vision_model(vision)

    out_counter_eeg_test_data_list.append(eeg_fea)
    out_counter_eeg_test_label_list.append(eeg_labels)

    out_counter_vis_test_data_list.append(vis_fea)
    out_counter_vis_test_label_list.append(vision_labels)


out_counter_eeg_test_data_list_tensor = torch.cat(out_counter_eeg_test_data_list,0)
print(f"EEG out-class test data shape is {out_counter_eeg_test_data_list_tensor.shape}")
out_counter_eeg_test_label_list_tensor = torch.cat(out_counter_eeg_test_label_list,0)
print(f"EEG out-class test label shape is {out_counter_eeg_test_label_list_tensor.shape}")

out_counter_vis_test_data_list_tensor = torch.cat(out_counter_vis_test_data_list,0)
print(f"E-MNIST out-class test data shape is {out_counter_vis_test_data_list_tensor.shape}")
out_counter_vis_test_label_list_tensor = torch.cat(out_counter_vis_test_label_list,0)
print(f"E-MNIST out-class test label shape is {out_counter_vis_test_label_list_tensor.shape}")


# print('==============================Begin train by the Readout ANN===========================================')

########## Data loader ############
train_dataset_feature = CustomTensorDataset_clip(
    tensors=(counter_eeg_train_data_list_tensor, counter_vis_train_data_list_tensor,
             counter_eeg_train_label_list_tensor, counter_vis_train_label_list_tensor),
    transform=None)

in_test_dataset_feature = CustomTensorDataset_clip(
    tensors=(in_counter_eeg_test_data_list_tensor, in_counter_vis_test_data_list_tensor,
             in_counter_eeg_test_label_list_tensor, in_counter_vis_test_label_list_tensor),
    transform=None)

out_test_dataset_feature = CustomTensorDataset_clip(
    tensors=(out_counter_eeg_test_data_list_tensor, out_counter_vis_test_data_list_tensor,
             out_counter_eeg_test_label_list_tensor, out_counter_vis_test_label_list_tensor),
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
    con_model = Contrastive_Simple(project_dim=options.pd, act_fun=options.act, temper_factor=options.temperature,
                            eeg_embedding=eeg_cfg_net[-2], vis_embedding=emnist_cfg_net[-2])
elif options.models == 'TwoFC':
    con_model = Contrastive_twolayers(project_dim=options.pd, temper_factor=options.temperature,
                            eeg_embedding=eeg_cfg_net[-2], vis_embedding=emnist_cfg_net[-2])
elif options.models == 'ThreeFC':
    con_model = Contrastive_threelayers(project_dim=options.pd, temper_factor=options.temperature,
                            eeg_embedding=eeg_cfg_net[-2], vis_embedding=emnist_cfg_net[-2])

con_model.to(device)

print(f"contrastive learning model is {con_model}")


optimizer = torch.optim.Adam(con_model.parameters(), lr=options.lr)

print(f'Feature batch size is',options_feature_batch)


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    for i, (f_images, f_audios, img_labels, aud_labels) in enumerate(valid_loader):

        images0 = Variable(f_images, requires_grad=True).to(device)
        audios = Variable(f_audios, requires_grad=True).to(device)

        loss = model(images0, audios, img_labels)
        count = images0.size(0)
        loss_meter.update(loss.item(), count)

    return loss_meter


def valid_acc(model,
              counter_eeg_test_data_list_tensor,
              counter_eeg_test_label_list_tensor,
              counter_vis_test_data_list_tensor,
              counter_vis_test_label_list_tensor,
              top=1):
    
    counter_eeg_test_data_list_tensor = counter_eeg_test_data_list_tensor.to(device)
    counter_vis_test_data_list_tensor = counter_vis_test_data_list_tensor.to(device)

    # eeg_embeddings = model.image_projection(counter_eeg_test_data_list_tensor)
    # vis_embeddings = model.audio_projection(counter_vis_test_data_list_tensor)

    eeg_embeddings = model.eeg_projection(counter_eeg_test_data_list_tensor)
    vis_embeddings = model.vis_projection(counter_vis_test_data_list_tensor)

    # normalized features
    eeg_embeddings_n = eeg_embeddings / eeg_embeddings.norm(dim=-1, keepdim=True)
    vis_embeddings_n = vis_embeddings / vis_embeddings.norm(dim=-1, keepdim=True)


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
    top_acc_v2e = count_m/len(counter_vis_test_label_list_tensor)*100.0

    # print(f"len(counter_vis_test_label_list_tensor) is {len(counter_vis_test_label_list_tensor)}, count_m is {count_m}")

    ### image search audio ########
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
    top_acc_e2v = count_aud / len(counter_eeg_test_label_list_tensor) * 100.0

    return top_acc_v2e, top_acc_e2v



best_acc_top1_v2e = 0.0
best_acc_top1_e2v = 0.0

cur_out_acc_v2e = 0.0
cur_out_acc_e2v = 0.0


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

    con_model.train()

    old_params = save_model_parameters(con_model)

    loss_meter = AvgMeter()
    for i, (f_images, f_audios, img_labels, aud_labels) in enumerate(train_feature_loader_crop):

        f_images = Variable(f_images, requires_grad=True).to(device)
        f_audios = Variable(f_audios, requires_grad=True).to(device)

        # print(f"image shape is {f_images.shape}")
        # print(f"image value is {f_images}")
        # print(f"audio shape is {f_audios.shape}")
        # print(f"audio value is {f_audios}")

        loss = con_model(f_images,f_audios, img_labels)
        # print(f"train loss is {loss.item()}")

        count = f_images.size(0)
        loss_meter.update(loss.item(), count)
                # con_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tqdm_object.set_postfix(train_loss=loss_meter.avg)

    new_params = save_model_parameters(con_model)
    
    print_parameters_diff(old_params, new_params)
    
    print(f"Train Loss is: {loss_meter.avg}")


    con_model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(con_model, in_test_feature_loader_crop)
        print(f"Test Loss is: {valid_loss.avg}")

        in_acc_top1_v2e, in_acc_top1_e2v = valid_acc(con_model,
                                               in_counter_eeg_test_data_list_tensor,
                                               in_counter_eeg_test_label_list_tensor,
                                               in_counter_vis_test_data_list_tensor,
                                               in_counter_vis_test_label_list_tensor,
                                               top=1)
        print(f"Top1 Accuracy for vision search eeg (in-classes): {in_acc_top1_v2e}")
        print(f"Top1 Accuracy for eeg search vision (in-classes): {in_acc_top1_e2v}")


        out_acc_top1_v2e, out_acc_top1_e2v = valid_acc(con_model,
                                               out_counter_eeg_test_data_list_tensor,
                                               out_counter_eeg_test_label_list_tensor,
                                               out_counter_vis_test_data_list_tensor,
                                               out_counter_vis_test_label_list_tensor,
                                               top=1)
        print(f"Top1 Accuracy for vision search eeg (out-classes): {out_acc_top1_v2e}")
        print(f"Top1 Accuracy for eeg search vision (out-classes): {out_acc_top1_e2v}")


        ### top1
    # if in_acc_top1_v2e >= 25 and out_acc_top1_v2e>=25:
    if in_acc_top1_v2e >= 0 and out_acc_top1_v2e >= 0:
        if in_acc_top1_v2e > best_acc_top1_v2e:
            best_acc_top1_v2e = in_acc_top1_v2e
            cur_out_acc_v2e = out_acc_top1_v2e

            state = {
                'eeg_model': eeg_model.state_dict(),
                'vis_model': vision_model.state_dict(),
                'clip_net': con_model.state_dict(),
                'in_acc': best_acc_top1_v2e,
                'out_acc': cur_out_acc_v2e
            }
            torch.save(state, f'{names}_top1_a2i.t7')
            print("Saved Best Model!")

            print('In-classes vision search eeg best accuracy is', best_acc_top1_v2e)
            print('Out-classes vision search eeg best accuracy is', cur_out_acc_v2e)


    ### top1
    if in_acc_top1_e2v >= 50 and out_acc_top1_e2v >=70:
    # if in_acc_top1_e2v >= 0 and out_acc_top1_e2v >= 0:
        if in_acc_top1_e2v > best_acc_top1_e2v:
            best_acc_top1_e2v = in_acc_top1_e2v
            cur_out_acc_e2v = out_acc_top1_e2v

            state = {
                'eeg_model': eeg_model.state_dict(),
                'vis_model': vision_model.state_dict(),
                'clip_net': con_model.state_dict(),
                'in_acc': best_acc_top1_e2v,
                'out_acc': cur_out_acc_e2v
            }
            torch.save(state, f'{names}_top1_i2a.t7')
            print("Saved Best Model!")

            print('In-classes eeg search vision best accuracy is', best_acc_top1_e2v)
            print('Out-classes eeg search vision best accuracy is', cur_out_acc_e2v)

print(f"Finally In Top1-acc for E-MNIST search EEG: {best_acc_top1_v2e}")
print(f"Finally In Top1-acc for EEG search E-MNIST: {best_acc_top1_e2v}")
print(f"Finally Out Top1-acc for E-MNIST search EEG: {cur_out_acc_v2e}")
print(f"Finally Out Top1-acc for EEG search E-MNIST: {cur_out_acc_e2v}")



