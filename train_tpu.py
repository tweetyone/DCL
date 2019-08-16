#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from transforms import transforms
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset

import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

import pdb
from math import ceil
import datetime
from utils.utils import LossRecord, clip_gradient
from utils.eval_model import eval_turn

# run instructions: 
# python train_tpu.py --data CUB --epoch 60 --backbone resnet50\
#                     --tb 16 --tnw 16 --vb 64 --vnw 16 \
#                     --lr 0.01 --lr_step 12 \
#                     --cls_lr_ratio 10 --start_epoch 0 \
#                     --detail training_descibe --size 512 \
#                     --crop 448 --cls_2 --swap_num 7 7

# if change the backbone to senet154, will have 'Unable to enqueue' error  


# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=16, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=512, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)

def dt():
    '''
    function to return current time
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def train(model,
          data_loader,
          device,
          context
          ):

    step = 0
    train_batch_size = args.train_batch
    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
        
    optimizer = context.getattr_or('optimizer', lambda: optim.SGD([{'params': base_params},
                         {'params': model.classifier.parameters(), 'lr': lr_ratio*base_lr},
                          {'params': model.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                          {'params': model.Convmask.parameters(), 'lr': lr_ratio*base_lr}     ], lr = base_lr, momentum=0.9))

    tracker = xm.RateTracker()
    
    model.train(True)

    for batch_cnt, data in enumerate(data_loader):
        step += 1
        loss = 0
        model.train(True)

        inputs, labels, labels_swap, swap_law, img_names = data[1]
        inputs = Variable(inputs)
        labels_1 = Variable(torch.from_numpy(np.array(labels)))
        labels_swap_1 = Variable(torch.from_numpy(np.array(labels_swap)))
        swap_law_1 = Variable(torch.from_numpy(np.array(swap_law)).float())
        # relocate tensor from cpu to tpu
        labels = labels_1.to(device)
        labels_swap = labels_swap_1.to(device)
        swap_law = swap_law_1.to(device)
            
        optimizer.zero_grad()

        if inputs.size(0) < 2*train_batch_size:
            outputs = model(inputs, inputs[0:-1:2])
        else:
            outputs = model(inputs, None)

        # calculate loss: alpha*ce_loss + beta*swap_loss + gamma*law_loss
        # ce_loss: classification loss
        # swap_loss: adversarial loss
        # law_loss: loc loss

        alpha_ = 1
        beta_ = 1
        gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
    
        add_loss = nn.L1Loss()
        get_ce_loss = nn.CrossEntropyLoss()

        ce_loss = get_ce_loss(outputs[0], labels) * alpha_
        loss += ce_loss
        swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
        loss += swap_loss
        law_loss = add_loss(outputs[2], swap_law) * gamma_
        loss += law_loss

        loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(train_batch_size)

        print('[{}] step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(device,step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), law_loss.detach().item()), flush=True)
    

if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    Config = LoadConfig(args, 'train')
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # initial dataloader
    # train_set aug+swap
    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["common_aug"],\
                        swap = transformers["swap"],\
                        totensor = transformers["train_totensor"],\
                        train = True)

    dataloader_train = torch.utils.data.DataLoader(train_set,\
                                                batch_size=args.train_batch,\
                                                shuffle=True,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=False,
                                                pin_memory=True)
    train_epoch_step = dataloader_train.__len__()

    setattr(dataloader_train, 'total_item_len', len(train_set))


    print('Choose model and train set', flush=True)
    model = MainModel(Config)

    # load model
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    time = datetime.datetime.now()

    num_cores = 8
    devices = (
        xm.get_xla_supported_devices(
            max_devices=num_cores) if num_cores != 0 else [])
    # Scale learning rate to num cores
    base_lr = args.base_lr * max(len(devices), 1)
    # Pass [] as device_ids to run using the PyTorch/CPU engine.
    model_parallel = dp.DataParallel(model, device_ids=devices)
   
    # optimizer prepare
    ignored_params1 = list(map(id, model.classifier.parameters()))
    ignored_params2 = list(map(id, model.classifier_swap.parameters()))
    ignored_params3 = list(map(id, model.Convmask.parameters()))

    ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())


    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)
    # exp_lr_scheduler.step(epoch)
    # train entry

    for epoch in range(1, args.epoch + 1):
        model_parallel(train, dataloader_train )





