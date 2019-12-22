#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:46:04 2019

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

def compute_detail_score(df, dice):
    res = []
    res.append(df[dice].mean())
    #c1 -> c4 dice
    for label in ['Fish', 'Flower', 'Gravel', 'Sugar']:
        df_tmp = df[df['cls'] == label]
        res.append(df_tmp[dice].mean())
    # neg & pos dice
    res.append(df[df['truth'] == ''][dice].mean())
    res.append(df[df['truth'] != ''][dice].mean())

    # c1 -> c4 pos
    for label in ['Fish', 'Flower', 'Gravel', 'Sugar']:
        df_tmp = df[df['cls'] == label]
        res.append(df_tmp[df_tmp['truth'] != ''][dice].mean())
    return res

def ensemble_rles(rles1, rles2, mode='intersect'):
    res = []
    for rle1, rle2 in tqdm.tqdm(zip(rles1, rles2)):
        m1 = rle2mask(rle1, height=350, width=525, fill_value=1)
        m2 = rle2mask(rle2, height=350, width=525, fill_value=1)
        if mode == 'intersect':
            mask = ((m1+m2) == 2).astype(int)
        elif mode == 'union':
            mask = ((m1+m2) > 0).astype(int)
        else:
            RuntimeError('%s not implemented.'%mode)
        rle = mask2rle(mask)
        res.append(rle)
    return res

df_train = pd.read_csv('../input/train_350.csv')
df_train.rename(columns={'EncodedPixels': 'truth'}, inplace=True)

fold=0
#seg1 = 'resnet34-FPN-BCE-warmRestart-bs16-t10'
seg1 = 'densenet121-FPN-BCE-warmRestart-bs16-t10'
seg1 = 'densenet121-FPN-BCE-warmRestart-bs16-PL'
#seg2 = 'efficientnetb5-Unet-DICE-warmRestart-bs16-t0'
seg2 = 'efficientnetb7-FPN-DICE-warmRestart-bs16-t0'
#classifier = 'resnet34-cls-BCE-HVSFR-W11-reduceLR'
classifier = 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0'

df_cls_val = pd.read_csv('../output/'+ classifier + '/' + 'valid_cls_fold%d.csv'%fold)
df_cls_test = pd.read_csv('../output/'+ classifier + '/' + 'test_cls_fold%d.csv'%fold)
df_cls_val.rename(columns={'0': 'prob'}, inplace=True)
df_cls_test.rename(columns={'0': 'prob'}, inplace=True)

df_seg1_val = pd.read_csv('../output/'+ seg1 + '/' + 'valid_fold%d.csv'%fold)
df_seg1_test = pd.read_csv('../output/'+ seg1 + '/' + 'test_fold%d.csv'%fold)

df_seg1_val.rename(columns={'EncodedPixels': 's1'}, inplace=True)
df_seg1_test.rename(columns={'EncodedPixels': 's1'}, inplace=True)


df_seg2_val = pd.read_csv('../output/'+ seg2 + '/' + 'valid_fold%d.csv'%fold)
df_seg2_test = pd.read_csv('../output/'+ seg2 + '/' + 'test_fold%d.csv'%fold)

df_seg2_val.rename(columns={'EncodedPixels': 's2'}, inplace=True)
df_seg2_test.rename(columns={'EncodedPixels': 's2'}, inplace=True)

df_seg_val = pd.merge(df_seg1_val, df_seg2_val, how='left')
df_seg_val = pd.merge(df_seg_val, df_train, how='left')
df_seg_val = pd.merge(df_seg_val, df_cls_val[['Image_Label', 'prob']], how='left')

df_seg_val['s3'] = df_seg_val['s1'].copy()
df_seg_val['s3'].loc[df_seg_val['s1'].notnull()] = df_seg_val['s2'].loc[df_seg_val['s1'].notnull()]
#df_seg_val['area'] = df_seg_val['s3'].apply(lambda x: rle2mask(x, height=350, width=525).sum())

df_seg_test = pd.merge(df_seg1_test, df_seg2_test, how='left')
df_seg_test = pd.merge(df_seg_test, df_cls_test[['Image_Label', 'prob']], how='left')
# Compute val score
df_seg_val.fillna('', inplace=True)
df_seg_val['cls'] = df_seg_val['Image_Label'].apply(lambda x: x.split('_')[-1])
df_seg_val['dice1'] = df_seg_val.apply(lambda x: dice_np_rle(x['s1'], x['truth']), axis=1)
df_seg_val['dice2'] = df_seg_val.apply(lambda x: dice_np_rle(x['s2'], x['truth']), axis=1)
df_seg_val['dice3'] = df_seg_val['dice1'].copy()
df_seg_val['dice3'].loc[df_seg_val['s1']!=''] = df_seg_val['dice2'].loc[df_seg_val['s1']!='']
#df_seg_val['dice3'] = df_seg_val.apply(lambda x: dice_np_rle(x['s3'], x['truth']), axis=1)

#print('Finding area size')
#area_ts = []
#best_dice_channel = []
#for label in ['Fish', 'Flower', 'Gravel', 'Sugar']:
#    df_tmp = df_seg_val[df_seg_val['cls'] == label].reset_index(drop=True).copy()
#    best_dice, best_ts = 0, 0
#    for area in tqdm.tqdm(range(0, 30000, 2000)):
#        df_tmp['s3'].loc[df_tmp['area'] <= area] = ''
#        df_tmp['dice'] = df_tmp.apply(lambda x: dice_np_rle(x['s3'], x['truth']), axis=1)
#        current_dice = np.mean(df_tmp['dice'])
#        if current_dice > best_dice:
#            best_dice = current_dice
#            best_ts = area
#    area_ts.append(best_ts)
#    best_dice_channel.append(best_dice)
#print(area_ts)
#print('Best dice: %.5f'%(np.mean(best_dice_channel)))

print('Finding cls threshold')
#cls_ts = []
#best_dice_channel = []
#df_seg_val['dice_cls'] = df_seg_val['dice3'].copy()
#for label in ['Fish', 'Flower', 'Gravel', 'Sugar']:
#    df_tmp = df_seg_val[df_seg_val['cls'] == label].reset_index(drop=True).copy()
#    best_dice, best_ts = 0, 0
#    for ts in tqdm.tqdm(range(0, 100, 1)):
#        ts /= 100
#        df_tmp['s3'].loc[df_tmp['prob'] <= ts] = ''
#        df_tmp['dice'] = df_tmp['dice_cls']
#        df_tmp['dice'].loc[(df_tmp['s3'] == '') & (df_tmp['truth'] =='')] = 1
#        df_tmp['dice'].loc[(df_tmp['s3'] != '') & (df_tmp['truth'] =='')] = 0
#        df_tmp['dice'].loc[(df_tmp['s3'] == '') & (df_tmp['truth'] !='')] = 0
#        current_dice = np.mean(df_tmp['dice'])
#        if current_dice > best_dice:
#            best_dice = current_dice
#            best_ts = ts
#    cls_ts.append(best_ts)
#    best_dice_channel.append(best_dice)
cls_ts = [0.5]*4
print(cls_ts)
#print('Best dice: %.5f'%(np.mean(best_dice_channel)))

df_seg_val['s4'] = df_seg_val['s3'].copy()
cls_mask = cls_ts * (df_seg_val.shape[0]//4)
df_seg_val['s4'].loc[df_seg_val['prob'] <= cls_mask] = ''
df_seg_val['dice4'] = df_seg_val['dice3'].copy()
df_seg_val['dice4'].loc[(df_seg_val['s4'] == '') & (df_seg_val['truth'] =='')] = 1
df_seg_val['dice4'].loc[(df_seg_val['s4'] != '') & (df_seg_val['truth'] =='')] = 0
df_seg_val['dice4'].loc[(df_seg_val['s4'] == '') & (df_seg_val['truth'] !='')] = 0
df_seg_val.to_csv('../output/ensemble/valid_fold%d_details.csv'%(fold), index=None)

print('                 dice   |  c1   |  c2   |  c3   |  c4   |  neg  |  pos  | pos1  | pos2  | pos3  | pos4')
res = compute_detail_score(df_seg_val, 'dice1')
res = ','.join('%.5f'%x for x in res)
print('Seg-1 scores:    %s'%res)
res = compute_detail_score(df_seg_val, 'dice2')
res = ','.join('%.5f'%x for x in res)
print('Seg-2 scores:    %s'%res)
res = compute_detail_score(df_seg_val, 'dice3')
res = ','.join('%.5f'%x for x in res)
print('ensemble scores: %s'%res)
res = compute_detail_score(df_seg_val, 'dice4')
res = ','.join('%.5f'%x for x in res)
print('     cls scores: %s'%res)
removed = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != ''))
tn = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != '') & (df_seg_val['truth'] == ''))
fn = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != '') & (df_seg_val['truth'] != ''))
print('Remove %d channels. TN: %d; FN: %d. ' % (removed, tn, fn))

df_seg_test['EncodedPixels'] = df_seg_test['s1'].copy()
df_seg_test['EncodedPixels'].loc[df_seg_test['s1'].notnull()] = df_seg_test['s2'].loc[df_seg_test['s1'].notnull()]
df_seg_test.to_csv('../output/ensemble/test_fold%d_details.csv'%(fold), index=None)
df_seg_test.to_csv('../output/ensemble/test_fold%d.csv'%(fold), index=None, columns=['Image_Label', 'EncodedPixels'])

cls_mask = cls_ts * (df_seg_test.shape[0]//4)
print('Remove %d channels.' % (np.sum((df_seg_test['prob'] <= cls_mask) & (df_seg_test['EncodedPixels'].notnull()))))
df_seg_test['EncodedPixels'].loc[df_seg_test['prob'] <= cls_mask] = ''
df_seg_test.to_csv('../output/ensemble/test_fold%d_cls_details.csv'%(fold), index=None)
df_seg_test.to_csv('../output/ensemble/test_fold%d_cls.csv'%(fold), index=None, columns=['Image_Label', 'EncodedPixels'])