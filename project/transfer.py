import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
import os
import json
import time
from sklearn import metrics

#local
from dataset import FMA
from models.simsiam import Evaluator
from utils import draw_curve, write_log, AverageMeter, get_confusion_matrix
from metrics import get_recall_precision
from parameters import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=64,type=int)
parser.add_argument('--size', default='small',type=str, choices=['small','medium'])
parser.add_argument('--aug', default='basic', type=str, choices=['basic','image'])
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=0.0001, type=float)


parser.add_argument('--exp_path', default='__?__', type=str,
                    help='./epx/??? <- ???넣어주세요')

parser.add_argument('--gpu_id', default='1', type=str)


args = parser.parse_args()


def transfer(train_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0

    for i, (audio, label) in enumerate(train_loader):        
        audio = audio.type(torch.float32).cuda()
        label = label.type(torch.LongTensor).cuda()

        optimizer.zero_grad()
        pred = model(audio)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            print(f'[ {i} | {len(train_loader)} ] Transfer_Train_loss : {np.around(loss.item(), 3)}')

        epoch_loss += loss.item()
    
    return epoch_loss


def validation(test_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0

        for i,(audio, label) in enumerate(test_loader):
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
            
            pred = model(audio)
            loss = criterion(pred, label)
            if i%10 == 0:
                print(f'[ {i} | {len(test_loader)} ] Transfer_val_loss : {np.around(loss.item(), 3)}')

            epoch_loss += loss.item()

    return epoch_loss


def test(test_loader, model):
    print('='*10 + ' Transfer Test ' + '='*10)
    
    pred_list = []
    label_list = []
    with torch.no_grad():
        model.eval()
        for audio, label in tqdm(test_loader):
            label_list += label.tolist()
            audio, label = audio.type(torch.float32).cuda(), label.type(torch.LongTensor).cuda()
    
            pred = model(audio)
            pred = torch.argmax(torch.sigmoid(pred),dim=-1)
            pred_list += pred.tolist()
            
    confusion_matrix = get_confusion_matrix(pred_list,label_list,8)
    recall_at_1, precision_at_1 = get_recall_precision(confusion_matrix)
    acc_at_1 = float(metrics.accuracy_score(label_list,pred_list))
    f1_at_1 = 2*recall_at_1*precision_at_1/(recall_at_1+precision_at_1+1e-3)

    result = {
              'Acc' : f'{acc_at_1:.3f}', 
              'Precision@1' : f'{precision_at_1:.3f}',
              'Recall@1' : f'{recall_at_1:.3f}',
              'F1@1' : f'{f1_at_1:.3f}'
              }

    # Save result, confusion matrix
    with open(f'./exp/{args.exp_path}/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    f = open(f'./exp/{args.exp_path}/confusion.txt','w')
    f.write(str(confusion_matrix))

    print(result)
    print(f'Confusion Matrix : \n{confusion_matrix}')
    print("=================== Test End ====================")
    



def main():
    ####### Envirement Setting #######
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    ####### Dataset #######
    train_data = FMA(split='validation',size=args.size)
    test_data = FMA(split='test',size=args.size)

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    ###### Load Pre-trained model #######
    with open(f'./exp/{args.exp_path}/configuration.json', 'r') as f: 
            configuration = json.load(f)
    # model = Siamusic(backbone=configuration['backbone'],
    #                  augmentation=configuration['aug'], 
    #                  dim=configuration['dim'], 
    #                  pred_dim=configuration['pred_dim'], 
    #                  sr=configuration['sr'], 
    #                  n_fft=configuration['n_fft'], 
    #                  f_min=configuration['f_min'], 
    #                  f_max=configuration['f_max'], 
    #                  n_mels=configuration['n_mels']).cuda()
    # model_parameters = torch.load(f'./exp/{args.exp_path}/pre.pth')
    # model.load_state_dict(model_parameters)
    model = torch.load(f'./exp/{args.exp_path}/pre.pth')

    transfer_model = Evaluator(encoder=model.encoder,
                               num_classes=8,
                               augmentation=configuration['aug'],
                               dim=configuration['dim'],
                               sample_rate=22050, 
                               n_fft=configuration['n_fft'], 
                               f_min=configuration['f_min'], 
                               f_max=configuration['f_max'], 
                               n_mels=configuration['n_mels']).cuda()


    ### freeze all layers except last fc layer ###
    for name, param in transfer_model.named_parameters():
        if name not in ['evaluator.0.weight', 'evaluator.0.bias', 'evaluator.2.weight', 'evaluator.2.bias',
        'evaluator.4.weight', 'evaluator.4.bias', 'evaluator.6.weight', 'evaluator.6.bias']:
            param.requires_grad = False
    
    # ### initialize the fc layer ###
    # for i in [0,2,4,6]:
    #     finetuned_model.evaluator[i].weight.data.normal_(mean=0.0, std=0.01)
    #     finetuned_model.evaluator[i].bias.data.zero_()
    # finetuned_model.cuda()

    criterion = nn.CrossEntropyLoss()

    parameters = list(filter(lambda p: p.requires_grad, transfer_model.parameters()))
    optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    milestones = [int(args.num_epochs/3), int(args.num_epochs/2)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

    best_loss = 10000.0
    transfer_train_loss_dic = {}
    transfer_val_loss_dic = {}

    for epoch in tqdm(range(args.num_epochs)):
        print(' ')
        start_time = time.time()

        print('='*10 + f' Transfer [{epoch+1}/{args.num_epochs} | eph/ephs] ' + '='*10)
        train_epoch_loss = transfer(train_loader, transfer_model, criterion, optimizer)
        train_epoch_loss = train_epoch_loss/len(train_loader)

        print('='*10 + f' Validation [{epoch+1}/{args.num_epochs} | eph/ephs] ' + '='*10)
        valid_epoch_loss = validation(test_loader, transfer_model, criterion)
        valid_epoch_loss = valid_epoch_loss/len(test_loader)
        
        transfer_train_loss_dic[epoch+1] = train_epoch_loss
        transfer_val_loss_dic[epoch+1] = valid_epoch_loss

        write_log(f'./exp/{args.exp_path}', 'transfer_train_log', str(epoch+1), np.around(train_epoch_loss, 3))
        write_log(f'./exp/{args.exp_path}', 'transfer_val_log', str(epoch+1), np.around(valid_epoch_loss, 3))

        print('='*10 + f' Epoch {epoch+1:02}: | Transfer CE Loss: {train_epoch_loss:.5f} | Validation CE Loss: {valid_epoch_loss:.5f} ' + '='*10)
        
        
        if valid_epoch_loss <= best_loss:
            best_loss = valid_epoch_loss
            torch.save(transfer_model.state_dict(), f'./exp/{args.exp_path}/transfer.pth')
            print(f'Saved At {epoch+1} epoch : Model saved!')
        
        scheduler.step()
    
    draw_curve(f'./exp/{args.exp_path}', transfer_train_loss_dic, 'transfer_train_loss')
    draw_curve(f'./exp/{args.exp_path}', transfer_val_loss_dic, 'transfer_val_loss')


    ####### Final Test #######
    transfer_model.load_state_dict(torch.load(f'./exp/{args.exp_path}/transfer.pth'))
    transfer_model.cuda()
    test(test_loader, transfer_model)
    print(f'========= It took {(time.time()-start_time)/60:.3f} mins ==========')

if __name__ == '__main__':
    # num_workers = 4 * torch.cuda.device_count()
    # test_data = FMA(split='test',size='small')
    # test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True, num_workers=num_workers)


    # with open('./exp/12/configuration.json', 'r') as f: 
    #         configuration = json.load(f)
    # model = torch.load(f'./exp/{args.exp_path}/pre.pth')

    # transfer_model = Evaluator(encoder=model.encoder,
    #                            num_classes=8,
    #                            augmentation=configuration['aug'],
    #                            dim=configuration['dim'],
    #                            sample_rate=22050, 
    #                            n_fft=configuration['n_fft'], 
    #                            f_min=configuration['f_min'], 
    #                            f_max=configuration['f_max'], 
    #                            n_mels=configuration['n_mels']).cuda()
    # transfer_model.load_state_dict(torch.load('./exp/12/transfer.pth'))
    # transfer_model.cuda()
    
    # test(test_loader, transfer_model)
    main()