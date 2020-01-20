# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:40:03
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:32:35



import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import tqdm
from tqdm import tqdm

from util.utils import *

DATA_ROOT = Path('../input/')
SIZE = 384

def resize_mask(size=SIZE):
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    H = size
    W = int(3/2*H)
    df.fillna('', inplace=True)
    for i in tqdm(range(df.shape[0])):
        rle = df['EncodedPixels'].values[i]
        if rle != '':
            mask = rle2mask(rle, height=1400, width=2100, fill_value=1)
            mask = (cv2.resize(mask, (W, H)) > 0).astype(int)
            new_rle = mask2rle(mask)
        else:
            new_rle = rle
        df['EncodedPixels'].iloc[i] = new_rle
    df.to_csv(DATA_ROOT/ ('train_%d.csv'%size), index=None)


def make_label_image(remove_class_2=False):
    train = pd.read_csv(DATA_ROOT / ('train_%d.csv'%SIZE))
    ref = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}
    train['defect'] = train['EncodedPixels'].notnull()
    train['ClassId'] = train['Image_Label'].apply(lambda x: ref[x.split('_')[-1]])
    train['ImageId'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train_df = train[['ImageId', 'ClassId', 'defect', 'EncodedPixels']]
    # train.to_csv(DATA_ROOT / 'class_label.csv',index=False)

    if remove_class_2:
        interval = 3
        train_df = train_df[train_df['ClassId'] != '2']
        train_df = train_df.reset_index(drop=True)
        file_name = 'class_label_without_2.csv'
    else:
        interval = 4
        file_name = 'class_label.csv'

    # train_df
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
                    temp_pixels.append(pixels[i])
            temp_label_str = ' '.join(str(t) for t in temp_label)
            temp_pixles_str = '|'.join(str(t) for t in temp_pixels)
            class_label['Labels'][col] = temp_label_str
            class_label['Is_defect'][col] = '1'
            class_label['EncodedPixels'][col] = temp_pixles_str

    new_df = pd.DataFrame.from_dict(class_label)
    new_df.to_csv(DATA_ROOT / file_name, index=False)


# build kFold

def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'class_label.csv')

    cls_counts = Counter(cls for classes in df['Labels'].str.split() for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                     total=len(df)):
        cls = min(item.Labels.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.Labels.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds

    return df


def main():
    resize_mask(SIZE)
    make_label_image(remove_class_2=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_csv(DATA_ROOT / ('5-folds_%d.csv'%(SIZE)), index=None)


if __name__ == '__main__':
    main()
