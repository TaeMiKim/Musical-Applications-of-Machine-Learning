import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = './dataset/'
feature_path = './feature/'

    
def summary_features(dataset='train'):
    
    feature_dict = {}
    
    for feature_name in os.listdir('./feature/'):

        first_file_name = os.listdir('./feature/{}/train'.format(feature_name))[0]
        first_file = np.load('./feature/{}/train/{}'.format(feature_name, first_file_name))

        if dataset == 'train':
            feature_mat_mean = np.zeros(shape=(first_file.shape[0], 1100))
#             feature_mat_var = np.zeros(shape=(first_file.shape[0], 1100))
        else:
            feature_mat_mean = np.zeros(shape=(first_file.shape[0], 300))
#             feature_mat_var = np.zeros(shape=(first_file.shape[0], 300))
              
        i = 0   
        f = open(data_path + dataset + '_list.txt','r')

        for file_name in f:

            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav','.npy')

            feature_file = feature_path + feature_name + '/' + file_name
            feature = np.load(feature_file)

            feature_mat_mean[:,i] = np.mean(feature, axis=1)
#             feature_mat_var[:,i] = np.std(feature, axis=1)

            i = i + 1

#         feature_dict[feature_name] = np.concatenate((feature_mat, feature_mat2))
        feature_dict[feature_name] = feature_mat_mean
  
    final_feature = np.concatenate(list(feature_dict.values()))
    
    return final_feature
        
if __name__ == '__main__':
    train_data = summary_features('train')
    valid_data = summary_features('valid')
