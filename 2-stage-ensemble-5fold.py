# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 1:27:19
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:01



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

def load_stacking(seg_name, tta, ts=0.5):
    df_seg_val = pd.read_csv('../output/'+seg_name+'/valid_5fold_tta%d.csv'%tta)
    df_seg_test = pd.read_csv('../output/'+seg_name+'/test_5fold_tta%d.csv'%tta)
    df_seg_val['s1'], df_seg_test['s1'] = np.nan, np.nan
    df_seg_val['s1'].loc[df_seg_val.pred >= ts] = '1 1'
    df_seg_test['s1'].loc[df_seg_test.pred >= ts] = '1 1'
    return df_seg_val[['Image_Label', 's1']], df_seg_test[['Image_Label', 's1']]

def load_seg_pred(seg_name, name, tta):
    #load val
    df_val = []
    try:
        for fold in range(5):
            if tta <= 1:
                df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_fold%d.csv'%fold))
            else:
                df_val.append(pd.read_csv('../output/'+ seg_name + '/' + 'valid_fold%d_tta%d.csv'%(fold, tta)))
        df_val = pd.concat(df_val)
    except:
        df_val = pd.read_csv('../output/'+ seg_name + '/' + 'valid_5fold_tta%d.csv'%(tta))
        df_val = df_val[['Image_Label', 'EncodedPixels']]
        #df_val.rename(columns={'s3': 'EncodedPixels'}, inplace=True)

    df_test = pd.read_csv('../output/'+ seg_name + '/' + 'test_5fold_tta%d.csv'%tta)
    df_val.rename(columns={'EncodedPixels': name}, inplace=True)
    df_test.rename(columns={'EncodedPixels': name}, inplace=True)
    return df_val, df_test

def load_seg_cls_pred(seg_name, name, tta, ts):
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
    df_val['EncodedPixels'] = '1 1'
    df_val['EncodedPixels'].loc[df_val['0'] < ts] = np.nan
    df_test['EncodedPixels'] = '1 1'
    df_test['EncodedPixels'].loc[df_test['0'] < ts] = np.nan
    df_val.rename(columns={'EncodedPixels': name}, inplace=True)
    df_test.rename(columns={'EncodedPixels': name}, inplace=True)
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

df_train = pd.read_csv('../input/train_350.csv')
df_train.rename(columns={'EncodedPixels': 'truth'}, inplace=True)

_save=1
tta=3

seg1 = 'densenet121-FPN-BCE-warmRestart-10x3-bs16'
seg2 = 'b5-Unet-inception-FPN-b7-Unet-b7-FPN-b7-FPNPL'
classifier = 'efficientnetb1-cls-BCE-reduceLR-bs16-PL'
# load classifier results
if classifier:
    if 'stacking' in classifier:
        df_cls_val = pd.read_csv('../output/'+classifier+'/valid_5fold_tta%d.csv'%tta).rename(columns={'pred': 'prob'})
        df_cls_test = pd.read_csv('../output/'+classifier+'/test_5fold_tta%d.csv'%tta).rename(columns={'pred': 'prob'})
    else:
        df_cls_val, df_cls_test = load_classifier(classifier, tta)

# load seg results
if isinstance(seg1, list):
    df_seg1_val, df_seg1_test = load_seg_pred(seg1[0], 's1', tta)
    for i in range(1, len(seg1)):
        d1, d2 = load_seg_pred(seg1[i], 's1', tta)
        df_seg1_val['s1'].loc[d1.s1.isnull()] = np.nan
        df_seg1_test['s1'].loc[d2.s1.isnull()] = np.nan
elif 'stacking' in seg1:
    df_seg1_val, df_seg1_test = load_stacking(seg1, 3, ts=0.54)
else:
    df_seg1_val, df_seg1_test = load_seg_pred(seg1, 's1', 1)

df_seg2_val, df_seg2_test = load_seg_pred(seg2, 's2', tta)
# merge seg valid
df_seg_val = pd.merge(df_seg1_val, df_seg2_val, how='left')
df_seg_val = pd.merge(df_seg_val, df_train, how='left')

if classifier:
    df_seg_val = pd.merge(df_seg_val, df_cls_val[['Image_Label', 'prob']], how='left')

df_seg_val['s3'] = df_seg_val['s1'].copy()
df_seg_val['s3'].loc[df_seg_val['s1'].notnull()] = df_seg_val['s2'].loc[df_seg_val['s1'].notnull()]
#df_seg_val['area'] = df_seg_val['s3'].apply(lambda x: rle2mask(x, height=350, width=525).sum())

df_seg_test = pd.merge(df_seg1_test, df_seg2_test, how='left')
df_seg_test = pd.merge(df_seg_test, df_cls_test[['Image_Label', 'prob']], how='left')
# Compute val score
print('Compute dice score.')
df_seg_val.fillna('', inplace=True)
df_seg_val['cls'] = df_seg_val['Image_Label'].apply(lambda x: x.split('_')[-1])
df_seg_val['dice1'] = df_seg_val.apply(lambda x: dice_np_rle(x['s1'], x['truth']), axis=1)
df_seg_val['dice2'] = df_seg_val.apply(lambda x: dice_np_rle(x['s2'], x['truth']), axis=1)
df_seg_val['dice3'] = df_seg_val['dice1'].copy()
df_seg_val['dice3'].loc[df_seg_val['s1']!=''] = df_seg_val['dice2'].loc[df_seg_val['s1']!='']

if classifier:
    print('Finding cls threshold for empty')
    cls_ts = []
    best_dice_channel = []
    df_seg_val['dice_cls'] = df_seg_val['dice3'].copy()
    for label in ['Fish', 'Flower', 'Gravel', 'Sugar']:
        df_tmp = df_seg_val[df_seg_val['cls'] == label].reset_index(drop=True).copy()
        best_dice, best_ts = 0, 0
        for ts in tqdm.tqdm(range(0, 100, 1)):
            ts /= 100
            df_tmp['s3'].loc[df_tmp['prob'] <= ts] = ''
            df_tmp['dice'] = df_tmp['dice_cls']
            df_tmp['dice'].loc[(df_tmp['s3'] == '') & (df_tmp['truth'] =='')] = 1
            df_tmp['dice'].loc[(df_tmp['s3'] != '') & (df_tmp['truth'] =='')] = 0
            df_tmp['dice'].loc[(df_tmp['s3'] == '') & (df_tmp['truth'] !='')] = 0
            current_dice = np.mean(df_tmp['dice'])
            if current_dice > best_dice:
                best_dice = current_dice
                best_ts = ts
        cls_ts.append(best_ts)
        best_dice_channel.append(best_dice)

    print(cls_ts)

    df_seg_val['s4'] = df_seg_val['s3'].copy()
    cls_mask = cls_ts * (df_seg_val.shape[0]//4)
    # remove empty
    df_seg_val['s4'].loc[df_seg_val['prob'] <= cls_mask] = ''

    df_seg_val['dice4'] = df_seg_val['dice3'].copy()
    df_seg_val['dice4'].loc[(df_seg_val['s4'] == '') & (df_seg_val['truth'] =='')] = 1
    df_seg_val['dice4'].loc[(df_seg_val['s4'] != '') & (df_seg_val['truth'] =='')] = 0
    df_seg_val['dice4'].loc[(df_seg_val['s4'] == '') & (df_seg_val['truth'] !='')] = 0
    df_seg_val['dice4'].loc[(df_seg_val['s4'] != '') & (df_seg_val['truth'] !='')] = \
    df_seg_val['dice2'].loc[(df_seg_val['s4'] != '') & (df_seg_val['truth'] !='')]

if _save:
    df_seg_val.to_csv('../output/ensemble/valid_5fold_tta%d.csv'%(tta), index=None)

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

if classifier:
    res = compute_detail_score(df_seg_val, 'dice4')
    res = ','.join('%.5f'%x for x in res)
    print('     cls scores: %s'%res)
    removed = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != ''))
    tn = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != '') & (df_seg_val['truth'] == ''))
    fn = np.sum((df_seg_val['prob'] <= cls_mask) & (df_seg_val['s3'] != '') & (df_seg_val['truth'] != ''))
    print('Remove %d channels. TN: %d; FN: %d. ' % (removed, tn, fn))

df_seg_test['EncodedPixels'] = df_seg_test['s1'].copy()
df_seg_test['EncodedPixels'].loc[df_seg_test['s1'].notnull()] = df_seg_test['s2'].loc[df_seg_test['s1'].notnull()]
if _save:
    df_seg_test.to_csv('../output/ensemble/test_5fold_tta%d_details.csv'%(tta), index=None)
    df_seg_test.to_csv('../output/ensemble/test_5fold_tta%d.csv'%(tta), index=None, columns=['Image_Label', 'EncodedPixels'])


if classifier:
    cls_mask = cls_ts * (df_seg_test.shape[0]//4)
    print('Test remove %d channels.' % (np.sum((df_seg_test['prob'] <= cls_mask) & (df_seg_test['EncodedPixels'].notnull()))))
    df_seg_test['EncodedPixels'].loc[df_seg_test['prob'] <= cls_mask] = np.nan
    if _save:
        df_seg_test.to_csv('../output/ensemble/test_5fold_tta%d_cls_details.csv'%(tta), index=None)
        df_seg_test.to_csv('../output/ensemble/test_5fold_tta%d_cls.csv'%(tta), index=None, columns=['Image_Label', 'EncodedPixels'])
