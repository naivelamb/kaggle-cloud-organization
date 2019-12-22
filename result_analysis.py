import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util.utils import *


def get_mask_size(rle):
    mask = rle2mask(rle, 1600, 256, 1)
    return np.sum(mask)

def get_cnt_info(classes, N_CLASSES=5):
    cls_cnt = []
    for i in range(N_CLASSES):
        cls_cnt.append(np.sum(classes == i))
    cnt_total = sum(cls_cnt)
    print('  neg,  pos1,  pos2,  pos3,  pos4')
    print(' %d,   %d,     %d,   %d,   %d'%(cls_cnt[0],cls_cnt[1],cls_cnt[2],
                                           cls_cnt[3],cls_cnt[4]))
    print('%.3f, %.3f, %.3f, %.3f, %.3f'%(cls_cnt[0]/cnt_total,cls_cnt[1]/cnt_total,
                                          cls_cnt[2]/cnt_total,cls_cnt[3]/cnt_total,
                                          cls_cnt[4]/cnt_total))

file = 'resnet34-FPN-BCE-HV-reduceLR-15x1-addNormalization'
linux = Path('../output').exists()
if linux:
    path = '../output/'
    df_train = pd.read_csv('../input/train_350.csv')
else:
    path = '../../KaggleSteel/'
    df_train = pd.read_csv('../../input/train_350.csv')
df_train.fillna('', inplace=True)

#size_thresholds = [150, 1100, 1100, 1100]
size_thresholds = [600, 600, 1100, 2000]
#size_thresholds = [0, 0, 0, 0]
##########################################
################valid part################
##########################################

df_val = pd.read_csv(path + '%s/valid_fold0.csv'%(file)).fillna('')
df_val.rename(columns={'EncodedPixels': 'pred'}, inplace=True)
df_val = pd.merge(df_val, df_train, how='left', on='Image_Label')
#df_val['area_pred'] = df_val['pred'].apply(get_mask_size)
#df_val_cls = pd.read_csv(path + '%s/valid_cls_all.csv'%(file))
#df_val_cls = pd.merge(df_val[['ImageId_ClassId']], df_val_cls, how='left', on='ImageId_ClassId')
#df_val['class_pred'] = df_val_cls['0']
#df_val['class'] = df_val['ImageId_ClassId'].apply(lambda x: int(x.split('_')[-1]))
#df_val['area'] = df_val['EncodedPixels'].apply(get_mask_size)

#df_val['class_all'] = df_val['class']
#df_val['class_all'].loc[df_val['area']==0] = 0
#
##seg
df_val['dice'] = df_val.apply(lambda x: dice_np_rle(x['pred'], x['EncodedPixels']), axis=1)
print(df_val.dice.mean())
#dice_details = [df_val['dice'].mean()]
#dice_details.append(df_val[df_val['area']==0]['dice'].mean())
#dice_details.append(df_val[df_val['area']!=0]['dice'].mean())
#for _cls in range(1, 5):
#    mask_cls = (df_val['class'] == _cls) & (df_val['area'] != 0)
#    dice_details.append(df_val[mask_cls]['dice'].mean())
#
#df_val['class_seg'] = 0
#df_val['class_seg'].loc[df_val['pred'] != ''] = 1
#df_val['class_seg'].loc[df_val['class_seg']>0.5] = df_val['class'].loc[df_val['class_seg']>0.5]
##cls
#df_val['class_cls'] = (df_val['class_pred'] > 0.5).astype(int)
#df_val['class_cls'].loc[df_val['class_pred']>0.5] = df_val['class'].loc[df_val['class_pred']>0.5]
#
#print(' ###### valid ######')
## valid ground truth
#get_cnt_info(df_val['class_all'].values)
## segmentation performance
## dice
#print('#########Segmentation#########')
#print('Dice:')
#print(' all  ,  neg  , pos   , pos1  , pos2  , pos3  , pos4')
#print(', '.join('%.4f'% x for x in dice_details))
## seg hit class
#print('Acc: %.4f' % (np.mean(df_val['class_all'] == df_val['class_seg'])))
#fig,ax,cm = plot_confusion_matrix(df_val['class_all'].values, df_val['class_seg'].values, [0,1,2,3,4])
#print('Normalized confusion matrix')
#print(np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3))
## classification performance
#print('#########Classifier#########')
## acc
#print('Acc: %.4f' % (np.mean(df_val['class_all'] == df_val['class_cls'])))
## cls hit class
#fig,ax,cm = plot_confusion_matrix(df_val['class_all'].values, df_val['class_cls'].values, [0,1,2,3,4])
#print('Normalized confusion matrix')
#print(np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3))
#print('Empty precision: %.3f' % (cm[0][0]/cm[:,0].sum()))
## post-processing
#df_val_tmp = df_val.copy()
#df_val_tmp['EncodedPixels'] = df_val['pred'].copy()
##Only small mask
#df_val_small = post_process(df_val_tmp.copy(), size_thresholds=size_thresholds)
#df_val_small['pred'] = df_val_small['EncodedPixels']
#df_val_small['EncodedPixels'] = df_val['EncodedPixels']
#df_val_small['area'] = df_val_small['pred'].apply(get_mask_size)
#df_val_small['dice'] = df_val_small.apply(lambda x: dice_np_rle(x['pred'], x['EncodedPixels']), axis=1)
#dice_details = [df_val_small['dice'].mean()]
#dice_details.append(df_val_small[df_val_small['area']==0]['dice'].mean())
#dice_details.append(df_val_small[df_val_small['area']!=0]['dice'].mean())
#for _cls in range(1, 5):
#    mask_cls = (df_val_small['class'] == _cls) & (df_val_small['area'] != 0)
#    dice_details.append(df_val_small[mask_cls]['dice'].mean())
#print('Post-process dice:\n' + ' all  ,  neg  , pos   , pos1  , pos2  , pos3  , pos4')
#print(', '.join('%.4f'% x for x in dice_details))
## Small mask + cls
#df_val_post = post_process(df_val_tmp, 
#                           df_val_cls,
#                           size_thresholds=size_thresholds
#                           )
#df_val_post['pred'] = df_val_post['EncodedPixels']
#df_val_post['EncodedPixels'] = df_val['EncodedPixels']
#df_val_post['area'] = df_val_post['pred'].apply(get_mask_size)
#df_val_post['dice'] = df_val_post.apply(lambda x: dice_np_rle(x['pred'], x['EncodedPixels']), axis=1)
#dice_details = [df_val_post['dice'].mean()]
#dice_details.append(df_val_post[df_val_post['area']==0]['dice'].mean())
#dice_details.append(df_val_post[df_val_post['area']!=0]['dice'].mean())
#for _cls in range(1, 5):
#    mask_cls = (df_val_post['class'] == _cls) & (df_val_post['area'] != 0)
#    dice_details.append(df_val_post[mask_cls]['dice'].mean())
#print('Post-process dice:\n' + ' all  ,  neg  , pos   , pos1  , pos2  , pos3  , pos4')
#print(', '.join('%.4f'% x for x in dice_details))
###########################################
#################pred part#################
###########################################
#df_sub = pd.read_csv(path + '%s/test_fold0.csv'%(file)).fillna('')
#df_sub_cls = pd.read_csv(path + '%s/test_cls_all.csv'%(file))
#df_sub['class_pred'] = df_sub_cls['0']
#df_sub['class'] = df_sub['ImageId_ClassId'].apply(lambda x: int(x.split('_')[-1]))
#df_sub['area'] = df_sub['EncodedPixels'].apply(get_mask_size)
#
#df_sub['class_seg'] = 0
#df_sub['class_seg'].loc[df_sub['EncodedPixels'] != ''] = 1
#df_sub['class_seg'].loc[df_sub['class_seg']>0.5] = df_sub['class'].loc[df_sub['class_seg']>0.5]
#df_sub['class_cls'] = 0
#df_sub['class_cls'].loc[df_sub['class_pred']>0.5] = df_sub['class'].loc[df_sub['class_pred']>0.5]
#print(' ###### prediction ###### ')
#print('LB probing results: ')
#print('  neg,  pos1,  pos2,  pos3,  pos4')
#print(' 6172,   128,    43,   741,   120')
#print('0.856, 0.017, 0.005, 0.102, 0.016')
#print('#########Segmentation#########')
## pred seg cnt
#get_cnt_info(df_sub['class_seg'].values)
#print('#########Classfier#########')
## pred cls cnt
#get_cnt_info(df_sub['class_cls'].values)
##post processing
#print('##### Post processing #####')
##df_cls_heng = pd.read_csv('../output/resnet34-cls.csv')
#df_sub_p = post_process(df_sub.copy(), 
#                        df_sub_cls,
#                        size_thresholds=size_thresholds
#                        )
#df_sub_p.to_csv(path + '%s/test_fold0_processed.csv'%(file),
#                columns=['ImageId_ClassId', 'EncodedPixels'],
#                index=None)
#df_sub_p['class_seg'] = 0
#df_sub_p['class_seg'].loc[df_sub_p['EncodedPixels'] != ''] = 1
#df_sub_p['class_seg'].loc[df_sub_p['class_seg']>0.5] = df_sub_p['class'].loc[df_sub_p['class_seg']>0.5]
#get_cnt_info(df_sub_p['class_seg'].values)
