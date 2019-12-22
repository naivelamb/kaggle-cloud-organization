#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:31:42 2019

@author: xuan
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

def load_seg_cls_pred(seg_name, name, tta):
    #load val
    df_val = []
    try:
        for fold in range(5):
            if tta <= 1:
                df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_cls_fold%d.csv'%fold))
            else:
                df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_cls_fold%d_tta%d.csv'%(fold, tta)))
        df_val = pd.concat(df_val)
    except:
        df_val = pd.read_csv('../output/'+ seg_name + '/' + 'valid_5fold_tta%d.csv'%(tta))
        df_val = df_val[['Image_Label', 'EncodedPixels']]
        #df_val.rename(columns={'s3': 'EncodedPixels'}, inplace=True)

    df_test = pd.read_csv('../output/'+ seg_name + '/' + 'test_cls_5fold_tta%d.csv'%tta)
#    df_val['EncodedPixels'] = '1 1'
#    df_val['EncodedPixels'].loc[df_val['0'] < ts] = np.nan
#    df_test['EncodedPixels'] = '1 1'
#    df_test['EncodedPixels'].loc[df_test['0'] < ts] = np.nan    
    df_val.rename(columns={'0': name}, inplace=True)
    df_test.rename(columns={'0': name}, inplace=True)
    return df_val, df_test

def load_classifier(classifier, tta):
    try:
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


        df_tmp = df_cls_test[0]
        for i in range(1, 5):
            assert(np.sum(df_tmp['Image_Label'] != df_cls_test[i]['Image_Label']) == 0)
            df_tmp['0'] += df_cls_test[i]['0']
        df_tmp['0'] /= 5
        df_cls_test = df_tmp
    except:
        df_cls_val = pd.read_csv('../output/'+ classifier + '/' + 'valid_cls_5fold_tta%d.csv'%tta)
        df_cls_test = pd.read_csv('../output/'+ classifier + '/' + 'test_cls_5fold_tta%d.csv'%tta)
    df_cls_val.rename(columns={'0': 'prob'}, inplace=True)
    df_cls_test.rename(columns={'0': 'prob'}, inplace=True)
    return df_cls_val, df_cls_test

df_train_raw = pd.read_csv('../input/train_350.csv')
df_train_raw.rename(columns={'EncodedPixels': 'truth'}, inplace=True)
df_train_raw['target'] = df_train_raw.truth.notnull().astype(int)
#df_train = df_train_raw[['Image_Label', 'truth', 'fold']].copy()
#df_train['target'] = df_train['truth'].notnull()

tta=3

#segs = [
#        'densenet121-FPN-BCE-warmRestart-bs16-t10',
#        'densenet121-Unet-BCE-reduceLR-10x3-bs16',
#        'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16',
#        'resnet34-JPU-BCE-warmRestart-bs16',
#        ]

clss = [
        'resnet34-cls-BCE-HVSFR-W11-reduceLR',
        'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0',
        'efficientnetb1-cls-BCE-HV-reduceLR-PL',
        'efficientnetb3-cls-BCE-HV-reduceLR'
        ]
#clss = [
#        
#        ]
#for i in range(len(segs)):
#    if i == 0:
#        df_train, df_test = load_seg_cls_pred(segs[i], 's%d'%(i+1), tta)
#        df_train = df_train[['Image_Label', 's1']]
#        df_test = df_test[['Image_Label', 's1']]
#    else:
#        d1, d2 = load_seg_cls_pred(segs[i], 's%d'%(i+1), tta)
#        df_train['s%d'%(i+1)] = d1['s%d'%(i+1)].values
#        df_test['s%d'%(i+1)] = d2['s%d'%(i+1)].values
for i in range(len(clss)):
    if i == 0:
        df_train, df_test = load_classifier(clss[i], tta)
        df_train = df_train[['Image_Label', 'prob']].rename(columns={'prob': 'prob1'})
        df_test = df_test[['Image_Label', 'prob']].rename(columns={'prob': 'prob1'})
    else:
        d1, d2 = load_classifier(clss[i], tta)
        df_train['prob%d'%(i+1)] = d1['prob'].values
        df_test['prob%d'%(i+1)] = d2['prob'].values

df_train = pd.merge(df_train, df_train_raw[['Image_Label', 'target', 'fold']], how='left')

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 5,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    #"bagging_fraction" : 0.4,
    #"feature_fraction" : 0.05,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : 42,
    "verbosity" : 1,
    "seed": 42
}

use_cols = [col for col in df_train.columns if col not in ['Image_Label', 'target', 'fold']]
y_p = np.zeros(df_test.shape[0])
df_vals = []
for i in range(5):
    df_trn = df_train[df_train['fold']!=i][use_cols]
    df_val = df_train[df_train['fold']==i][use_cols]
    y_trn = df_train[df_train['fold']!=i]['target']
    y_val = df_train[df_train['fold']==i]['target']
    trn_data = lgb.Dataset(df_trn, label=y_trn)
    val_data = lgb.Dataset(df_val, label=y_val)
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                    trn_data,
                    100000,
                    valid_sets = [trn_data, val_data],
                    early_stopping_rounds=200,
                    verbose_eval = 100,
                    evals_result=evals_result
                   )
    
    y_p += lgb_clf.predict(df_test[use_cols])/5
    p_val = lgb_clf.predict(df_val)
    df_val_tmp = df_train[df_train['fold']==i][['Image_Label']].reset_index(drop=True)
    df_val_tmp['pred'] = p_val
    df_vals.append(df_val_tmp)

df_vals = pd.concat(df_vals)
df_res = df_test[['Image_Label']]
df_res['pred'] = y_p


df_vals.to_csv('../output/valid_5fold_tta%d.csv'%tta, index=None)
df_res.to_csv('../output/test_5fold_tta%d.csv'%tta, index=None)

df_vals = pd.merge(df_train[['Image_Label', 'target']], df_vals, how='left')

p_ts = 0.54
index0 = (df_vals.pred < p_ts)
index1 = (df_vals.pred >= p_ts)

tn = np.sum(df_vals[index0].target==0)
tp = np.sum(df_vals[index1].target==1)
acc = (tn+tp)/df_vals.shape[0]
tnr = tn/np.sum(df_vals.target==0)
tpr = tp/np.sum(df_vals.target==1)
print('Acc: %.5f. TNR: %.5f. TPR: %.5f.' % (acc, tnr, tpr))