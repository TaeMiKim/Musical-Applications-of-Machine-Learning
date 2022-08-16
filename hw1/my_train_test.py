# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from my_feature_summary import *

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_linear_SVM(train_X, train_Y, valid_X, valid_Y, hyper_param1):

    # Choose a classifier (here, linear SVM)
    clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    

    # train
    clf.fit(train_X, train_Y)
    

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    
    print('alpha:' + str(hyper_param1) + ' --- validation accuracy = ' + str(accuracy) + ' %')
    
    return clf, accuracy


def train_nonlinear_SVM(train_X, train_Y, valid_X, valid_Y, gamma):

    # Choose a classifier (here, linear SVM)
    clf = SVC(kernel='rbf', gamma=gamma)
    

    # train
    clf.fit(train_X, train_Y)
    

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    
    print('gamma:' + str(gamma) + ' --- validation accuracy = ' + str(accuracy) + ' %')
    
    return clf, accuracy


def train_MLP(train_X, train_Y, valid_X, valid_Y, size):

    # Choose a classifier (here, linear SVM)
    clf = MLPClassifier(hidden_layer_sizes=(size))
    

    # train
    clf.fit(train_X, train_Y)
    

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    
    print('hidden_layer:', size, ' --- validation accuracy = ' + str(accuracy) + ' %')
    
    return clf, accuracy


if __name__ == '__main__':

    # load data 
    train_X = summary_features('train')
    valid_X = summary_features('valid')

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    alphas = [0.0001, 0.001 , 0.01]
    gammas = [0.001, 0.01, 0.1]
    hidden_layer_sizes = [i for i in range(45,125)]

    model = []
    valid_acc = []
#     for a in alphas:
#         clf, acc = train_linear_SVM(train_X, train_Y, valid_X, valid_Y, a)
#         model.append(clf)
#         valid_acc.append(acc)
        
#     for g in gammas:
#         clf, acc = train_nonlinear_SVM(train_X, train_Y, valid_X, valid_Y, g)
#         model.append(clf)
#         valid_acc.append(acc)
    
#     for size in hidden_layer_sizes:
#         clf, acc = train_MLP(train_X, train_Y, valid_X, valid_Y, size)
#         model.append(clf)
#         valid_acc.append(acc)
    
    clf, acc = train_MLP(train_X, train_Y, valid_X, valid_Y, 115)
    valid_acc.append(acc)
    
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('final validation accuracy = ' + str(accuracy) + ' %')

