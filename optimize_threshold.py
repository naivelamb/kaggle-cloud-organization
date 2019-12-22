#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:34:05 2019

@author: x0c02jg
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from util.utils import *

from tqdm import tqdm
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--file', default='resnet34-FPN-BCE-HV-reduceLR(1e-3)')
arg('--tta', type=int, default=1)
args = parser.parse_args()

file = args.file
tta = args.tta
print('Processing %s' % (file))
linux = Path('../output').exists()
if linux:
    path = '../output/'
    df_train = pd.read_csv('../input/train_350.csv')
else:
    path = '../../cloud/output/'
    df_train = pd.read_csv('../../cloud/train_350.csv')
df_train.fillna('', inplace=True)
df_train.rename(columns={'EncodedPixels': 'truth'}, inplace=True)


area_thresholds = [x for x in range(0, 30000, 2000)]
pixel_thresholds = [x/100 for x in range(0, 80, 5)]

param_pairs = []
for pixel_ts in pixel_thresholds:
    for area_ts in area_thresholds:
        param_pairs.append((pixel_ts, area_ts))

def compute_score(pairs, path=path, file=file, tta=tta):
    ans = []

    for pixel_ts, area_ts in tqdm(pairs):
        dfs = []
        for fold in range(5):
            file_name = 'opt_fold%d_tta%d_[%s].csv'%(fold, tta, str(pixel_ts))
            dfs.append(pd.read_csv(path + file + '/opt/' + file_name))
        dfs = pd.concat(dfs).reset_index(drop=True)
        dfs = pd.merge(dfs, df_train, how='left')
        dfs.fillna('', inplace=True)
        dfs['EncodedPixels'] = dfs['EncodedPixels'].apply(lambda x: str(x))
        dfs['area'] = dfs['EncodedPixels'].apply(lambda x: rle2mask(x, height=350, width=525).sum())
        dfs['EncodedPixels'].loc[dfs['area'] <= area_ts] = ''
        dfs['dice'] = dfs.apply(lambda x: dice_np_rle(x['EncodedPixels'], x['truth']), axis=1)
        dice_channel = np.reshape(dfs['dice'].values, (-1, 4))
        dice_channel = np.mean(dice_channel, axis=0).tolist()
        ans.append([pixel_ts, area_ts] + dice_channel)
    ans = pd.DataFrame(ans, columns=['pixel', 'area', '0', '1', '2', '3'])
    return ans


#def compute_score(pixel_ts, path=path, file=file, tta=tta, area_thresholds=area_thresholds):
#    all_res = {}
#    for pix_ts in pixel_ts:
#        scores = {}
#
#        for area_ts in tqdm(area_thresholds):
#
#        all_res[pix_ts] = scores
#    return all_res

n_cores = 12

pool = mp.Pool(n_cores)
n_cnt = len(param_pairs) // n_cores

dfs = [param_pairs[n_cnt*i:n_cnt*(i+1)] for i in range(n_cores)]
dfs[-1] = param_pairs[n_cnt*(n_cores-1):]
res = pool.map(compute_score, [x_df for x_df in dfs])
pool.close()


df_res = pd.concat(res)
df_res.to_csv(path+file+'/opt_res_tta%d.csv'%(tta), index=None)


best_area = []
best_pixel = []
best_dice = []
for i in range(4):
    scores = df_res[str(i)].values
    idx = np.where(scores == scores.max())
    best_pixel.append(df_res['pixel'].values[idx[0]][0])
    best_area.append(df_res['area'].values[idx[0]][0])
    best_dice.append(scores.max())

print('Area: ', best_area)
print('Pixel: ', best_pixel)
print('Dice: [%s]' % (','.join(['%.3f'%x for x in best_dice])))
print('Avg Dice: %.5f' % np.mean(best_dice))
