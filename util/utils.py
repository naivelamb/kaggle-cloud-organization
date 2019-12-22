# @Author: Xuan Cao <xuan>
# @Date:   2019-10-27, 4:32:34
# @Last modified by:   xuan
# @Last modified time: 2019-10-30, 11:07:50



from datetime import datetime
import json
import glob
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import Dict, Tuple
from math import ceil

from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import cv2

import torch.nn.functional as F

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

def set_seed(seed=6750):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def gmean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).agg(lambda x: gmean(list(x)))


def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()


def load_model(model: nn.Module, path: Path, multi2single=False) -> Tuple:
    state = torch.load(str(path), map_location='cuda:0')
    if multi2single:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state, state['best_valid_loss']

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr

    return True

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def write_event(log, step: int, epoch=None, **data):
    data['step'] = step
    data['dt'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['epoch']=epoch
    log.write(json.dumps(data, sort_keys=True, cls=MyEncoder))
    log.write('\n')
    log.flush()

def _smooth(ys, indices):
    return [np.mean(ys[idx: indices[i + 1]])
            for i, idx in enumerate(indices[:-1])]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          show_fig=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    #if normalize:
        #cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3)
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    if show_fig:
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        #fig.tight_layout()
        return fig, ax, cm
    else:
        return None, None, cm

def dice_channel_torch(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_by_channel = [0, 0, 0, 0]
    neg, pos, pos1, pos2, pos3, pos4 = 0, 0, 0, 0, 0, 0,
    n_neg, n_pos, n_pos1, n_pos2, n_pos3, n_pos4 = 0, 0, 0, 0, 0, 0,
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_by_channel[j] += channel_dice
                if truth[i, j, :, :].sum().cpu().numpy() > 0:# pos
                    pos += channel_dice
                    n_pos += 1
                    if j == 0:
                        pos1 += channel_dice
                        n_pos1 += 1
                    elif j == 1:
                        pos2 += channel_dice
                        n_pos2 += 1
                    elif j == 2:
                        pos3 += channel_dice
                        n_pos3 += 1
                    elif j == 3:
                        pos4 += channel_dice
                        n_pos4 += 1
                else:
                    neg += channel_dice
                    n_neg += 1
    return mean_dice_by_channel, [neg,pos,pos1,pos2,pos3,pos4], [n_neg,n_pos,n_pos1,n_pos2,n_pos3,n_pos4]

def dice_single_channel(probability, truth, threshold, eps = 1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def dice_np_rle(rle, rle_truth, height=350, width=525, threshold=0.5):
    if rle_truth == '' or rle == '':
        if rle_truth == '' and rle == '':
            return 1
        else:
            return 0
    else:
        pred = rle2mask(rle, height=height, width=width)
        truth = rle2mask(rle_truth, height=height, width=width)
        return dice_single_channel_np(pred, truth, threshold=threshold)

def dice_single_channel_np(pred, truth, threshold, eps = 1E-9):
    p = (pred.reshape(-1) > threshold)
    t = (truth.reshape(-1) > 0.5)
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def metric_hit(cls_pred, cls_targets, threshold=0.5):
    index0 = (cls_targets==0)
    index1 = (cls_targets[:, 0]==1)
    index2 = (cls_targets[:, 1]==1)
    index3 = (cls_targets[:, 2]==1)
    index4 = (cls_targets[:, 3]==1)

    p = cls_pred > threshold
    tn = (p[index0] == 0).sum()
    pos1 = (p[:, 0][index1] == 1).sum()
    pos2 = (p[:, 1][index2] == 1).sum()
    pos3 = (p[:, 2][index3] == 1).sum()
    pos4 = (p[:, 3][index4] == 1).sum()

    neg_precision = tn/((p==0).sum())
    neg_recall = tn/(index0.sum())

    aucs = []
    for i in range(4):
        aucs.append(roc_auc_score(cls_targets[:, i].reshape(-1), cls_pred[:, i].reshape(-1)))
    return [tn, pos1, pos2, pos3, pos4], aucs, [neg_precision, neg_recall]

def search_f1(output, target):
    max_result_f1_list = []
    max_threshold_list = []
    precision_list = []
    recall_list = []
    eps=1e-20

    # print(output.shape, target.shape)
    for i in range(output.shape[1]):

        output_class = output[:, i]
        target_class = target[:, i]
        max_result_f1 = 0
        max_threshold = 0

        optimal_precision = 0
        optimal_recall = 0

        for threshold in [x * 0.01 for x in range(0, 100)]:

            prob = output_class > threshold
            label = target_class
            # print(prob, label)
            TP = (prob & label).sum()
            TN = ((~prob) & (~label)).sum()
            FP = (prob & (~label)).sum()
            FN = ((~prob) & label).sum()

            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            # print(precision, recall)
            result_f1 = 2 * precision  * recall / (precision + recall + eps)

            if result_f1 > max_result_f1:
                # print(max_result_f1, max_threshold)
                max_result_f1 = result_f1
                max_threshold = threshold

                optimal_precision = precision
                optimal_recall = recall

        max_result_f1_list.append(round(max_result_f1,3))
        max_threshold_list.append(max_threshold)
        precision_list.append(round(optimal_precision,3))
        recall_list.append(round(optimal_recall,3))

    return max_threshold_list, max_result_f1_list, precision_list, recall_list

def get_current_consistency_weight(epoch, consistency=10, consistency_rampup=5.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def mask2rle(mask):  # 1:53
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = mask.T.flatten()
    if pixels.sum() == 0:
        rle = ''
    else:
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        rle = ' '.join(str(x) for x in runs)
    return rle

def mask2rle350(img):
    img = (cv2.resize(img, (525, 350)) > 0).astype(int)
    return mask2rle(img)

def rle2mask(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height, width), np.float32)
    if rle != '':
        mask = mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start, length in r:
            start = start - 1  # ???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask = mask.reshape(width, height).T
    return mask

def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)

def build_sub(image_ids, probs, mode, thresholds=[0.5, 0.5, 0.5, 0.5],
              area_size=[0, 0, 0, 0]):
    rles = []
    ref = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}
    for image_id, mask in zip(image_ids, probs):
        _class_name = image_id.split('_')[-1]
        _class = ref[_class_name]
        p_threshold = thresholds[_class]
        area_threshold = area_size[_class]
        mask = np.array(mask > p_threshold, dtype='uint8')
        if np.sum(mask):
            if mode == 'test' or 'valid' or 'opt' or 'predict_5fold':
                mask = (cv2.resize(mask, (525, 350)) > 0).astype(int)
                if mask.sum() >= area_threshold:
                    rles.append(mask2rle(mask))
                else:
                    rles.append('')
            else:
                rles.append(mask2rle(mask))
        else:
            rles.append('')

    df = pd.DataFrame({
                       'Image_Label': image_ids,
                       'EncodedPixels': rles
                       })
    return df

def remove_small_mask(probability, threshold, min_size,):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    H, W, _ = probability.shape
    predictions = np.zeros((H, W), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def post_process(df_sub, df_cls=None, size_thresholds=[200, 1500, 1500, 2000]):
    #[100, 250, 100, 400] from min size
    rles = []
    df_sub.fillna('', inplace=True)
    modified = [0]*4
    for image_id, rle in zip(df_sub['ImageId_ClassId'].tolist(), df_sub['EncodedPixels'].tolist()):
        _class = int(image_id.split('_')[-1])
        min_size = size_thresholds[_class-1]
        mask = rle2mask(rle)
        mask, _ = remove_small_mask(mask, 0.5, min_size)
        rle_new = mask2rle(mask)
        if rle_new != rle:
            modified[_class-1] += 1
        rles.append(rle_new)
    print('Modified small masks: ', modified)
    df_sub['EncodedPixels'] = rles

    if isinstance(df_cls, pd.DataFrame):
        to_remove = ((df_sub['EncodedPixels'] != '') & (df_cls['EncodedPixels'].isnull()))
        df_sub['EncodedPixels'].loc[to_remove] = ''
        print('Removed %d mask based on classification' % (np.sum(to_remove)))
    return df_sub

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def pad_tensor(img, target_size):
    """Pad an tensor up to the target size.
    :param img: (N, C, H, W)
    :param target_size: (H, W)
    :returns padded_img: (N, C, target_size[0], target_size[1])
    """
    (N, C, H, W) = img.size()
    (t_H, t_W) = target_size

    padded_img = torch.zeros(N, C, t_H, t_W)
    padded_img[:, :, :H, :W] = img

    return padded_img

def predict_sliding(model, image, tile_size=(256, 256), num_class=4, overlap=1 / 3, cls=True):
    """Use sliding window sized at tile_size to predict and average predictions
    :param model: nn.Module
    :param image: NCHW Tensor
    :param tile_size: padded sliding window size to fit the model input
    :param num_classes: class number
    :param overlap: controls sliding window step size
    :return: NCHW tensor as probability
    """
    (batch_size, _, H, W) = image.size()
    stride = ceil(tile_size[0] * (1 - overlap))  # 256*(1-1/3) = 170
    tile_rows = int(ceil((H - tile_size[0]) / stride) + 1)  # 行滑动步数:(256-256)/170 + 1 = 1
    tile_cols = int(ceil((W - tile_size[1]) / stride) + 1)  # 列滑动步数:(1600-256)/170 + 1 = 9
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = torch.zeros(
        batch_size, num_class, H, W,
        dtype=torch.float32,
        device=get_device()
    )  # 初始化全概率矩阵 shape(N, 256,1600,4)

    count_predictions = torch.zeros(
        batch_size, num_class, H, W,
        dtype=torch.float32,
        device=get_device()
    )  # 初始化计数矩阵 shape(N, 256,1600,4)

    for row in range(tile_rows):  # row = 0
        for col in range(tile_cols):  # col = 0,1,2,3,4,5,6,7,8
            x1 = int(col * stride)  # 起始位置x1 = 0 * 170 = 0
            y1 = int(row * stride)  # y1 = 0 * 170 = 0
            x2 = min(x1 + tile_size[1], W)  # 末位置x2 = min(0+256, 1600)
            y2 = min(y1 + tile_size[0], H)  # y2 = min(0+256, 256)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(256-256, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(256-256, 0)

            patch = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:256, 0:256]
            padded_patch = (pad_tensor(patch, tile_size)).cuda()  # padding确保扣下来的图像为256*256
            if cls:
                patch_pred, _ = model(padded_patch)
            else:
                patch_pred = model(padded_patch)
            prediction = patch_pred[:, :, 0:patch.shape[2], 0:patch.shape[3]]  # 扣下相应面积 shape(256,256,4)
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += prediction  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 平均概率
    return full_probs, torch.zeros(batch_size, num_class).cuda()

def predict_sliding_cls(model, image, tile_size=(256, 400), num_class=4, overlap=0):
    """Use sliding window sized at tile_size to predict and average predictions
    :param model: nn.Module
    :param image: NCHW Tensor
    :param tile_size: padded sliding window size to fit the model input
    :param num_classes: class number
    :param overlap: controls sliding window step size
    :return: NCHW tensor as probability
    """
    (batch_size, _, H, W) = image.size()
    stride = ceil(tile_size[1] * (1 - overlap))  # 400*(1-0) = 400
    tile_rows = int(ceil((H - tile_size[0]) / stride) + 1)  # 行滑动步数:(256-256)/400 + 1 = 1
    tile_cols = int(ceil((W - tile_size[1]) / stride) + 1)  # 列滑动步数:(1600-400)/400 + 1 = 4
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = torch.zeros(
        batch_size, num_class,
        dtype=torch.float32,
        device=get_device()
    )  # 初始化全概率矩阵 shape(batch_size, 4)

    for row in range(tile_rows):  # row = 0
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 400 = 0
            y1 = int(row * stride)  # y1 = 0 * 400 = 0
            x2 = min(x1 + tile_size[1], W)  # 末位置x2 = min(0+400, 1600)
            y2 = min(y1 + tile_size[0], H)  # y2 = min(0+256, 256)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(400-400, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(256-256, 0)

            patch = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:256, 0:400]
            padded_patch = (pad_tensor(patch, tile_size)).cuda()  # padding确保扣下来的图像为256*400
            patch_pred = model(padded_patch).float()

            if row == 0 and col == 0:
                full_probs = patch_pred
            else:
                full_probs = torch.max(full_probs, patch_pred)

    return full_probs
