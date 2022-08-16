# library
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

### Loss ###
class CosineLoss(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim

    def neg_cos_sim(self,p,z):
        z = z.detach()
        p = F.normalize(p,dim=self.dim) # default : L2 norm
        z = F.normalize(z,dim=self.dim)
        return -torch.mean(torch.sum(p*z,dim=self.dim))
    
    def forward(self,p1,z2,p2,z1):
        L = self.neg_cos_sim(p1,z2)/2 + self.neg_cos_sim(p2,z1)/2
        return L

### metric ###
def get_confusion_matrix(pred,target,num_classes):
    '''
    pred : [0,1,1,4,2,...]
    target : [0,1,3,4,2,...]
    '''
    confusion_matrix = torch.zeros((num_classes, num_classes))
    
    for i in range(len(pred)):
        confusion_matrix[pred[i],target[i]] += 1
    return confusion_matrix



### learning scheduler ###
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


### logging ###
def write_log(save_path, name, epoch, loss):
    with open(save_path + '/' + name + '.txt', 'a') as f:
        f.write(str(epoch) + '   ' + str(loss) + '\n')


def draw_curve(work_dir, epoch_loss_dic, name):
    epoch_list = []
    loss_list = []
    for epoch, loss in epoch_loss_dic.items():
        epoch_list.append(epoch)
        loss_list.append(loss)
    
    plt.plot(epoch_list, loss_list, color='red', label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(work_dir + '/' + name + '.png')
    plt.close()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass