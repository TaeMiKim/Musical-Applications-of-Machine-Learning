from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import librosa
import os
import torch


genre_dic_small = np.load('./data/genre_dic_small.npy', allow_pickle=True).item()
genre_dic_medium = np.load('./data/genre_dic_medium.npy', allow_pickle=True).item()
'''
FMA Small에서 098565, 098567, 098569는 모두 지워주세요(길이가 안 맞는 노이즈입니다)
'''

class FMA(Dataset):
    def __init__(self, split, size='small', input_length=22050*5):
        assert split in ['training', 'validation', 'test']
        assert size in ['small', 'medium']
        self.split = split #training, validation, test
        self.input_length = input_length
        self.size = size

        self.audio_path = f'./data/fma_{size}_npy'
        self.df = pd.read_csv(f'./data/tracks_{size}.csv')
        self.tracks_df = self.df[self.df['split'] == self.split] 
    
    def __len__(self):
        return len(self.tracks_df)
        # return 100

    def __getitem__(self, idx):
        genre = self.tracks_df.iloc[idx]['genre_top']
        if self.size == 'small':
            label = np.array(genre_dic_small[genre])
        else:
            label = np.array(genre_dic_medium[genre])

        id = self.tracks_df.iloc[idx]['track_id']
        audio_id = '0'*(6-len(str(id))) + str(id)
        audio = np.load(f'{self.audio_path}/{audio_id}.npy', allow_pickle=True).item()['audio']
        audio_len = audio.shape[0]
        try:
            start = random.randint(0, audio_len-self.input_length)
            end = start + self.input_length
            audio = audio[start:end]
        except:
            audio = audio[0:self.input_length]

        return audio, label # (B, 110250), (B)




if __name__ == '__main__':
    dataloader = DataLoader(FMA('training'),batch_size=1,shuffle=False)
    audio, label = next(iter(dataloader))
    print(audio.shape) # (B, 110250)
    print(label.shape) # (B)