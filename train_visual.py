#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:38:12 2019

@author: xuan
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = 'efficientnetb1-cls-BCE-HV-reduceLR-PL'
NUM_CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

for i in range(5):
    file_name = '../output/' + file + '/' + 'train-%d.log'%(i)
    try:
        df = pd.read_csv(file_name, sep='|')
        cols = df.columns
        df.columns = [x.strip() for x in cols]
        fig, ax = plt.subplots(2, 2, figsize=(12,12))
        #loss profile
        ax[0, 0].plot(df.epoch, df.loss, label='train-loss', marker='o')
        ax[0, 0].plot(df.epoch, df['val loss'], label='val-loss', marker='x')
        ax[0, 0].set_xlabel('epoch')
        ax[0, 0].set_ylabel('loss')
        ax[0, 0].legend()
        #lr profile
        ax[0, 1].plot(df.epoch, df.lr, label='lr', marker='o')
        ax[0, 1].set_xlabel('epoch')
        ax[0, 1].set_ylabel('lr')
        ax[0, 1].legend()   
        if 'AUC-mean' in df.columns: #cls
            ax[1, 0].plot(df.epoch, df['AUC-mean'], '-ro', label='AUC-mean')
            ax[1, 0].set_xlabel('epoch')
            ax[1, 0].set_ylabel('AUC-mean')
            ax[1, 0].legend()            
            for k in range(4):
                ax[1, 1].plot(df.epoch, df['class%d'%(k+1)], '-o', label=NUM_CLASSES[k])
            ax[1, 1].set_xlabel('epoch')
            ax[1, 1].set_ylabel('AUC')
            ax[1, 1].legend() 
        else:
            ax[1, 0].plot(df.epoch, df['val dice'], '-ro', label='dice')
            ax[1, 0].set_xlabel('epoch')
            ax[1, 0].set_ylabel('val-dice')
            ax[1, 0].legend()
        fig.savefig('../output/' + file + '/' + 'train_%d.png'%(i))
    except:
        pass
    