from h11 import Data
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

from parameters import sr, n_fft, f_min, f_max, n_mels
from dataset import FMA
from torch.utils.data import DataLoader


'''
이하 Audio Augmentation
'''

#lambda 값이 매번 random하게 변화
def mixup_random_lambda(x, alpha=0.4, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    if lam < 0.5:
        lam = 0.5

    batch_size = x.shape[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x


#lambda 값 고정
def mixup_fixed_lambda(x, lam=0.6, use_cuda=True):
    batch_size = x.shape[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x


def pitch_shift(batch, n_steps, sr=22050):
    batch_shifted = librosa.effects.pitch_shift(batch, sr=sr, n_steps=n_steps)
    
    return batch_shifted


def time_stretch(batch, rate):
    batch_stretched = librosa.effects.time_stretch(batch, rate=rate)
    padded_samples = np.zeros(shape=batch.shape)
    window = batch_stretched[..., : batch.shape[-1]]
    actual_window_length = window.shape[-1] 
    padded_samples[..., :actual_window_length] = window
    fixed_length_batch = padded_samples

    return fixed_length_batch


def train_audio_aug(batch, lam=None, sr=22050, p=0.4):
    n_steps = random.randint(-4, 4) # pitch_shift hyperparameter
    rate = random.uniform(0.6, 1.5) # time_stretch hyperparameter

    mix_p = random.random()   # mixup 적용 확률
    pitch_p = random.random() # pitch_shift 적용 확률
    time_p =  random.random() # time_stretch 적용 확률
    
    #확률이 hyperparameter p를 넘으면 각 augmentation 적용되도록 함. 
    if mix_p > p:
        # batch = mixup_fixed_lambda(batch, lam=lam, use_cuda=True)
        batch = mixup_random_lambda(batch, use_cuda=True)
    
    if pitch_p > p:
        batch = pitch_shift(batch.cpu().numpy(), n_steps=n_steps, sr=sr)
        batch = torch.from_numpy(batch)
    
    if time_p > p:
        batch = time_stretch(batch.cpu().numpy(), rate=rate)
        batch = torch.from_numpy(batch)

    return batch



'''
이하 Spectrogram Augmentation
'''

spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, f_min=f_min, f_max=f_max, n_mels=n_mels)
to_db = torchaudio.transforms.AmplitudeToDB()
spec_bn = nn.BatchNorm2d(1)


def maskout(batch, lam=0.4, use_cuda=True):
    '''
    lam : 잘라서 버릴 최대 비율
    '''
    ratio = random.uniform(0.1, lam)

    if use_cuda:
        mask = torch.ones((batch.shape)).cuda()
    else:
        mask = torch.ones((batch.shape))

    if random.random() > 0.5: # 행 mask
        mask_length = int(batch.shape[1]*ratio)
        start = random.randint(0, batch.shape[1]-mask_length-1)
        mask[:,start:start+mask_length,:,:] = 0
    else: # 열 mask
        mask_length = int(batch.shape[2]*ratio/3)
        a = random.randint(0, batch.shape[2]-mask_length-1)
        b = random.randint(0, batch.shape[2]-mask_length-1)
        c = random.randint(0, batch.shape[2]-mask_length-1)
        mask[:,:,a:a+mask_length,:] = 0
        mask[:,:,b:b+mask_length,:] = 0
        mask[:,:,c:c+mask_length,:] = 0

    masked_batch = mask * batch
    return masked_batch


def cutmix(batch, lam=0.3, use_cuda=True):
    '''
    lam : 섞여 들어갈 음원이 가질 수 있는 최대비율
    '''
    batch_size = batch.shape[0]
    length = batch.shape[2]
    ratio = random.uniform(0.1, lam)
    mix_length = int(length*ratio/3)
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        mask = torch.ones((batch.shape)).cuda()
    else:
        index = torch.randperm(batch_size)
        mask = torch.ones((batch.shape))
    
    a = random.randint(0, length-mix_length-1)
    b = random.randint(0, length-mix_length-1)
    c = random.randint(0, length-mix_length-1)
    
    mask[:,:,a:a+mix_length,:] = 0.0
    mask[:,:,b:b+mix_length,:] = 0.0
    mask[:,:,c:c+mix_length,:] = 0.0

    if use_cuda:
        mixed_batch = mask * batch + (torch.ones((batch.shape)).cuda() - mask) * batch[index]
    else:
        mixed_batch = mask * batch + (torch.ones((batch.shape)) - mask) * batch[index]

    return mixed_batch



def mixup(batch, lam=0.3, use_cuda=True):
    batch_size = batch.shape[0]
    ratio = random.uniform(0.1,lam)

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_batch = (1 - ratio) * batch + ratio * batch[index]
    return mixed_batch



def train_image_aug(batch, p=0.6, use_cuda=False):
    if use_cuda:
        spec.cuda()
        to_db.cuda()
        spec_bn.cuda()
    batch = to_db(spec(batch))
    batch = spec_bn(torch.unsqueeze(batch,1))

    batch = torch.permute(batch,(0,2,3,1))

    maskout_p = random.random()
    cutmix_p = random.random()
    mixup_p = random.random()

    if p > maskout_p:
        batch = maskout(batch, use_cuda=use_cuda)
    
    if p > cutmix_p:
        batch = cutmix(batch, use_cuda=use_cuda)

    if p > mixup_p:
        batch = mixup(batch, use_cuda=use_cuda)

    
    
    strong_aug = A.Compose([A.HorizontalFlip(p=0.8),
                            A.VerticalFlip(p=0.8),
                            A.GridDistortion(p=0.8),
                            A.Cutout(num_holes=20,max_h_size=5,max_w_size=5,p=0.8)])
    for idx in range(len(batch)):
        batch[idx] = torch.from_numpy(strong_aug(image=batch[idx].cpu().detach().numpy())['image'])

    
    return batch



if __name__ == '__main__':
    dataloader = DataLoader(FMA('training'),batch_size=12)
    audio, label = next(iter(dataloader))
    audio = audio.cuda()
    audio_1 = train_audio_aug(audio) # (B, 22050*5)
    audio_2 = train_image_aug(audio) # (B, 48, 431, 1)