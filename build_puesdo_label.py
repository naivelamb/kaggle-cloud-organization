#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:56:55 2019

@author: xuan
"""

import pandas as pd
import numpy as np
import cv2
from collections import defaultdict, Counter
from util.utils import *

def select_sample(df_test, ts_neg, ts_pos):
    neg = (df_test.EncodedPixels.isnull()&(df_test.prob < ts_neg))
    pos = (df_test.EncodedPixels.notnull()&(df_test.prob > ts_pos))
    df_test['pick'] = (neg | pos).astype(int)
    return df_test.pick.values

test_file = '11-11_2Seg_cls_5fold_tta13_LB6790'
tta=1
ts_neg, ts_pos = 0.3, 0.7

df_test = pd.read_csv('../output/ensemble/' + test_file + '/test_5fold_tta%d_cls_details.csv'%tta)

pick_idx = select_sample(df_test.copy(), ts_neg, ts_pos)
print('Channels: %d' % (pick_idx.sum()))

pick_idx = pick_idx.reshape(-1, 4)
pick_images = (pick_idx.sum(axis=1) == 4).astype(int)
pick_images = np.array([pick_images]*4).transpose()
pick_images = pick_images.reshape(-1)
print('Select images: %d' % (pick_images.sum()//4))

df_select = df_test[['Image_Label', 'EncodedPixels']].loc[pick_images==1].reset_index(drop=True)

ref = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}
df_select['defect'] = df_select['EncodedPixels'].notnull()
df_select['ClassId'] = df_select['Image_Label'].apply(lambda x: ref[x.split('_')[-1]])
df_select['ImageId'] = df_select['Image_Label'].apply(lambda x: x.split('_')[0])
train_df = df_select[['ImageId', 'ClassId', 'defect', 'EncodedPixels']]

# train_df
interval=4
class_label = defaultdict(dict)
for col in range(0, len(train_df), interval):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + interval, 0].values]

    defects = train_df.iloc[col:col + interval, 2]
    Ids = train_df.iloc[col:col + interval, 1]
    pixels = train_df.iloc[col:col + interval, 3]

    class_label['Image_Label'][col] = (img_names[0])
    temp_label, temp_pixels = [], []
    if pixels.isna().all():
        class_label['Labels'][col] = '0'
        class_label['Is_defect'][col] = '0'
        class_label['EncodedPixels'][col] = None
    else:
        for i in range(col, col + interval):
            if defects[i]:
                temp_label.append(Ids[i])
                pixel = pixels[i]
                # resize mask from (350x525) to (384x576)
                mask = rle2mask(pixel, height=350, width=525, fill_value=1)
                mask = (cv2.resize(mask, (576, 384)) > 0).astype(int)
                rle = mask2rle(mask)
                temp_pixels.append(rle)
        temp_label_str = ' '.join(str(t) for t in temp_label)
        temp_pixles_str = '|'.join(str(t) for t in temp_pixels)
        class_label['Labels'][col] = temp_label_str
        class_label['Is_defect'][col] = '1'
        class_label['EncodedPixels'][col] = temp_pixles_str

new_df = pd.DataFrame.from_dict(class_label)
new_df.to_csv('../input/df_pl.csv', index=None)