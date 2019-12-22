# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 1:25:27
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:27



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

def ensemble_rles_multi(rles, ts=0):
    res = []
    for rle in tqdm.tqdm(rles):
        m = np.zeros((350, 525))
        for x in rle:
            m += rle2mask(x, height=350, width=525, fill_value=1)

        mask = (m > ts).astype(int)

        ans = mask2rle(mask)
        res.append(ans)
    return res

def load_seg_pred(seg_name, name, tta):
    #load val
    df_val = []
    for fold in range(5):
        if tta <= 1:
            df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_fold%d.csv'%fold))
        else:
            df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_fold%d_tta%d.csv'%(fold, tta)))
    df_val = pd.concat(df_val)

    df_test = pd.read_csv('../output/'+ seg_name + '/' + 'test_5fold_tta%d.csv'%tta)
    df_val.rename(columns={'EncodedPixels': name}, inplace=True)
    df_test.rename(columns={'EncodedPixels': name}, inplace=True)
    return df_val, df_test

df_train = pd.read_csv('../input/train_350.csv')
df_train.rename(columns={'EncodedPixels': 'truth'}, inplace=True)

_save=1
tta=3
ts=2

ensemble_file_name = 'b5-Unet-inception-FPN-b7-Unet-b7-FPN-b7-FPNPL'
segs = [
        'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16',      #0.66823, 0.67019
        'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16', #0.67133, 0.67311
        'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16',         #0.67060, 0.67268
        'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16',      #0.67070, 0.67253
        'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16-pl', #0.67332, 0.67511
        ]

# load seg results
for i in range(len(segs)):
    df1, df2 = load_seg_pred(segs[i], 's%d'%(i+1), tta)
    if i == 0:
        df_seg_val, df_seg_test = df1, df2
    else:
        df_seg_val['s%d'%(i+1)] = df1['s%d'%(i+1)]
        df_seg_test['s%d'%(i+1)] = df2['s%d'%(i+1)]

## merge seg valid
df_seg_val = pd.merge(df_seg_val, df_train, how='left')

df_seg_val.fillna('', inplace=True)
df_seg_val['s%d'%(len(segs)+1)] = ensemble_rles_multi(df_seg_val[['s%d'%(i+1) for i in range(len(segs))]].values.tolist(), ts)

# Compute val score
print('Compute dice score.')
df_seg_val.fillna('', inplace=True)
df_seg_val['cls'] = df_seg_val['Image_Label'].apply(lambda x: x.split('_')[-1])
for i in range(len(segs)):
    df_seg_val['dice%d'%(i+1)] = df_seg_val.apply(lambda x: dice_np_rle(x['s%d'%(i+1)], x['truth']), axis=1)

df_seg_val['dice%d'%(len(segs)+1)] = df_seg_val.apply(lambda x: dice_np_rle(x['s%d'%(len(segs)+1)], x['truth']), axis=1)

if _save:
    df_seg_val.rename(columns={'s%d'%(len(segs)+1): 'EncodedPixels'}, inplace=True)
    df_seg_val.to_csv('../output/%s/valid_5fold_tta%d.csv'%(ensemble_file_name, tta), index=None)

print('')
print('                 dice   |  c1   |  c2   |  c3   |  c4   |  neg  |  pos  | pos1  | pos2  | pos3  | pos4')
for i in range(len(segs)):
    res = compute_detail_score(df_seg_val, 'dice%d'%(i+1))
    res = ','.join('%.5f'%x for x in res)
    print('Seg-%d scores:    %s'%(i+1, res))

res = compute_detail_score(df_seg_val, 'dice%d'%(len(segs)+1))
res = ','.join('%.5f'%x for x in res)
print('ensemble scores: %s'%res)

if _save:
    df_seg_test.fillna('', inplace=True)
    df_seg_test['EncodedPixels'] = ensemble_rles_multi(df_seg_test[['s%d'%(i+1) for i in range(len(segs))]].values.tolist(), ts)
    df_seg_test.to_csv('../output/%s/test_5fold_tta%d_details.csv'%(ensemble_file_name, tta), index=None)
    df_seg_test.to_csv('../output/%s/test_5fold_tta%d.csv'%(ensemble_file_name, tta), index=None, columns=['Image_Label', 'EncodedPixels'])
