import sys
import os
import numpy as np
import librosa
import argparse

data_path = './dataset/'
feature_path = './feature/'

parser = argparse.ArgumentParser()

parser.add_argument("--n_mfcc", default=20, type=int)
parser.add_argument("--n_mels", default=40, type=int)


def extract_features(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=22050)


        ##### mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=args.n_mfcc, n_mels=args.n_mels)
        
        ##### mfcc delta
        mfcc_delta = librosa.feature.delta(mfcc)
        
        ##### mfcc double-delta
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        ##### mfcc with constant-Q transform
        mfcc_CQT = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'), n_bins=args.n_mfcc))
        mfcc_delta_CQT = librosa.feature.delta(mfcc_CQT)
        mfcc_delta2_CQT = librosa.feature.delta(mfcc_CQT, order=2)
        
        ##### spectral statistics
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        ##### zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        ##### Chroma
#         chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        ##### temporal envelope
#         oenv = librosa.onset.onset_strength(y=y, sr=sr)
#         tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr)

        # save features as a file
        file_name = file_name.replace('.wav','.npy')
        
        features = [mfcc, mfcc_delta, mfcc_delta2, mfcc_CQT, mfcc_delta_CQT, mfcc_delta2_CQT, 
                    spec_centroid, spec_bw, spec_contrast, spec_flatness, spec_rolloff, zcr, 
#                     chroma, tempogram
                   ]
        
        features_name = ['mfcc', 'mfcc_delta', 'mfcc_delta2', 'mfcc_CQT', 'mfcc_delta_CQT', 'mfcc_delta2_CQT', 
                    'spec_centroid', 'spec_bw', 'spec_contrast', 'spec_flatness', 'spec_rolloff', 'zcr',
#                          'chroma', 'tempogram'
                        ]
        
        
        for feature, feature_name in zip(features, features_name):
            save_file = feature_path + feature_name + '/' + file_name
            
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            
            np.save(save_file, feature)

    f.close();

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    extract_features(dataset='train')                 
    extract_features(dataset='valid')                                  

