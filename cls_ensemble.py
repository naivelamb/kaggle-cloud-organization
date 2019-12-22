#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:33:44 2019

@author: xuan
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from util.utils import *

def load_classifier(classifier, tta):
    df_cls_val = []
    df_cls_test = []
    for fold in range(5):
        if tta <= 1:
            df_cls_val.append(pd.read_csv('../output/'+ classifier + '/' + 'valid_cls_fold%d.csv'%fold))
            df_cls_test.append(pd.read_csv('../output/'+ classifier + '/' + 'test_cls_fold%d.csv'%fold))
        else:
            df_cls_val.append(pd.read_csv('../output/'+ classifier + '/' + 'valid_cls_fold%d_tta%d.csv'%(fold, tta)))
            df_cls_test.append(pd.read_csv('../output/'+ classifier + '/' + 'test_cls_fold%d_tta%d.csv'%(fold, tta)))
    df_cls_val = pd.concat(df_cls_val)
    df_cls_val.rename(columns={'0': 'prob'}, inplace=True)

    df_tmp = df_cls_test[0]
    for i in range(1, 5):
        assert(np.sum(df_tmp['Image_Label'] != df_cls_test[i]['Image_Label']) == 0)
        df_tmp['0'] += df_cls_test[i]['0']
    df_tmp['0'] /= 5
    df_cls_test = df_tmp
    df_cls_test.rename(columns={'0': 'prob'}, inplace=True)
    return df_cls_val, df_cls_test

df_train = pd.read_csv('../input/train_350.csv')
df_train.rename(columns={'EncodedPixels': 'truth'}, inplace=True)

_save=1
tta=1
classifiers = [
        'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0',
        'resnet34-cls-BCE-HVSFR-W11-reduceLR',
        'efficientnetb3-cls-BCE-HV-reduceLR',
        ]

for i in range(len(classifiers)):
    classifier = classifiers[i]
    if i == 0:
        df_val, df_test = load_classifier(classifier, tta)
        df_val.rename(columns={'prob': 'p%d'%i}, inplace=True)
        df_test.rename(columns={'prob': 'p%d'%i}, inplace=True)
    else:
        df1, df2 = load_classifier(classifier, tta)
        df_val['p%d'%(i)] = df1['prob']
        df_test['p%d'%(i)] = df2['prob']
df_val['0'] = df_val[['p%d'%(i) for i in range(len(classifiers))]].mean(axis=1)
df_test['0'] = df_test[['p%d'%(i) for i in range(len(classifiers))]].mean(axis=1)

if _save:
    df_val.to_csv('../output/valid_cls_5fold_tta%d.csv'%(tta), index=None)
    df_test.to_csv('../output/test_cls_5fold_tta%d.csv'%(tta), index=None)

df_val = pd.merge(df_val, df_train[['Image_Label', 'truth']], how='left', on='Image_Label')
df_val['y'] = df_val['truth'].notnull().astype(int)
for i in range(len(classifiers)):
    acc = np.mean((df_val['p%d'%i]>=0.5) == df_val['y'])
    tp = np.sum(df_val[df_val['p%d'%i]>=0.5]['y'] == 1)
    tn = np.sum(df_val[df_val['p%d'%i]<0.5]['y'] == 0)
    print('%s acc: %.5f; TN: %d, TP: %d.' % (classifiers[i], acc, tn, tp))
acc = np.mean((df_val['0']>=0.5) == df_val['y'])
tp = np.sum(df_val[df_val['0']>=0.5]['y'] == 1)
tn = np.sum(df_val[df_val['0']<0.5]['y'] == 0)
print('Ensemble acc: %.5f; TN: %d, TP: %d.' % (acc, tn, tp))