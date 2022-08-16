from torch.utils.data import DataLoader
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import time
import json
import torchaudio

# local
from utils import adjust_learning_rate, write_log, draw_curve
from dataset import FMA
from utils import CosineLoss
from models.simsiam import Siamusic
from augmentation import train_audio_aug, train_image_aug
from parameters import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--size', default='small', type=str, choices=['small', 'medium'])
parser.add_argument('--optim', default='SGD', type=str, help='SGD or Adam')
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--backbone', default='resnet50', type=str, help='Select the model among resnet50, resnet101, resnet152')
parser.add_argument('--dim', default=2048, type=int)
parser.add_argument('--pred_dim', default=512, type=int)
parser.add_argument('--sr', default=22050, type=int)
parser.add_argument('--n_fft', default=512, type=int)
parser.add_argument('--f_min', default=0.0, type=float)
parser.add_argument('--f_max', default=8000.0, type=float)
parser.add_argument('--n_mels', default=80, type=int)

parser.add_argument('--aug', default='basic', choices=['basic', 'image'])
parser.add_argument('--lam', default=0.6, help='mixup hyperparameter')
parser.add_argument('--aug_p', default=0.6, help='augmenatation probability')

parser.add_argument('--gpu_id', default='1', type=str)

args = parser.parse_args()



def train(train_loader, model, criterion, optimizer):
    model.train()
    train_epoch_loss = 0
    

    for i,(audio, label) in enumerate(train_loader):
        if args.aug == 'basic':
            audio_aug1 = train_audio_aug(audio, p=args.aug_p)
            audio_aug2 = train_audio_aug(audio, p=args.aug_p)
        elif args.aug == 'image':
            audio_aug1 = train_image_aug(audio, p=args.aug_p)
            audio_aug2 = train_image_aug(audio, p=args.aug_p)
            audio_aug1 = torch.permute(audio_aug1,(0,3,1,2))
            audio_aug2 = torch.permute(audio_aug2,(0,3,1,2))
        audio_aug1, audio_aug2 = audio_aug1.type(torch.float32).cuda(), audio_aug2.type(torch.float32).cuda()

        optimizer.zero_grad() 
        p1, z2, p2, z1 = model(audio_aug1, audio_aug2)

        train_loss = criterion.cuda()(p1, z2, p2, z1)
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()

        if i%10 == 0:
            print(f'[ {i} | {len(train_loader)} ] Train_loss : {np.around(train_loss.item(), 3)}')
    return train_epoch_loss



def main():
    ####### Save Path #######
    current_time = str(time.time()).split('.')[-1]
    save_path = './exp/' + current_time

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ####### Configuration #######
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ####### GPU enviorement ######
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ####### dataloader setting #######
    train_data = FMA(split='training',size=args.size)

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    ####### model setting #######
    train_model = Siamusic(backbone=args.backbone,
                           augmentation=args.aug, 
                           dim=args.dim, 
                           pred_dim=args.pred_dim, 
                           sr=args.sr, 
                           n_fft=args.n_fft, 
                           f_min=args.f_min, 
                           f_max=args.f_max, 
                           n_mels=args.n_mels).cuda()
    
   

    ####### loss setting #######
    train_criterion = CosineLoss()
    

    ####### optimizer setting #######
    init_lr = args.lr * args.batch_size / 256
    
    if args.fix_pred_lr:
        optim_params = [{'params': train_model.encoder.parameters(), 'fix_lr': False},
                        {'params': train_model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = train_model.parameters()

    if args.optim == 'SGD':
        optimizer = optim.SGD(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(optim_params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ####### Pre-training ########
    best_loss = 10000.0
    best_epoch = 0
    count = 0
    train_loss_dic = {}

    
    for epoch in tqdm(range(args.num_epochs)):
        print(' ')
        start_time = time.time()
        
        print('='*10 + f' Training [{epoch+1}/{args.num_epochs} | eph/ephs] ' + '='*10)
        train_epoch_loss = train(train_loader, train_model, train_criterion, optimizer)
        train_epoch_loss = train_epoch_loss/len(train_loader)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        write_log(save_path, 'train_log', str(epoch+1), np.around(train_epoch_loss, 3))
        train_loss_dic[epoch+1] = train_epoch_loss

        print(f'========== Epoch {epoch+1} of {args.num_epochs}: | Train Cosine Loss: {train_epoch_loss:.5f} | It took {(time.time()-start_time)/60:.3f} mins ==========')
        
        
        if train_epoch_loss > best_loss:
            count += 1
            if (epoch>30) and (count==5):
                print('='*20)
                print(f'Early Stopping At {epoch+1} epoch : The model saved at {best_epoch+1}epoch lastly!')
                break
            else:
                pass
        elif train_epoch_loss <= best_loss:
            count = 0
            best_loss = train_epoch_loss
            best_epoch = epoch
            # torch.save(train_model.state_dict(), f'{save_path}/pre.pth')
            torch.save(train_model, f'{save_path}/pre.pth')
            print('='*20)
            print(f'Saved At {epoch+1} epoch : Model saved!')


    draw_curve(save_path, train_loss_dic, 'train_loss')

if __name__ == '__main__':
    main()
