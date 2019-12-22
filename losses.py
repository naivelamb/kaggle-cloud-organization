# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 1:32:21
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:21



from common import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import segmentation_models_pytorch as smp

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(-1, 1)
        target = target.view(-1, 1)

        pt = torch.sigmoid(input)
        pt = 1 - (pt - target.float()).abs()
        logpt = pt.log()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.long().data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()# -*- coding: utf-8 -*-

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-9
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2*intersection + eps)/(union + eps)
        score = (1 - score).mean()
        return score

class WeightedBCE(nn.Module):
    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit, truth):
        batch_size, num_class = truth.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert(logit.shape == truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if self.weights is None:
            loss = loss.mean()
        else:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.weights[1]*pos*loss/pos_sum + self.weights[0]*neg*loss/neg_sum).sum()
        return loss

def multiclass_dice_loss(logits, targets):
    loss = 0
    dice = SoftDiceLoss()
    num_classes = targets.size(1)
    for class_nr in range(num_classes):
        loss += dice(logits[:, class_nr, :, :], targets[:, class_nr, :, :])
    return loss/num_classes

def combo_loss(logits, fc=0, labels=0, labels_fc=0, weights=[0.1, 0, 1], activation=None, per_image=0):
    # weights -> [image_cls, pixel_seg, pixel_cls]
    # image class
    if activation == 'sigmoid':
        p_labels = F.sigmoid(logits)
    elif activation is None:
        p_labels = logits
    else:
        RuntimeError('%s activation not implemented' % (activation))
    if weights[0]:
        loss_fc = weights[0] * nn.BCEWithLogitsLoss(reduce=True)(fc, labels_fc)
    else:
        loss_fc = torch.tensor(0).cuda()
    if weights[1] or weights[2]:
        # pixel seg
        if per_image:
            loss_seg_dice = weights[1] * SoftDiceLoss()(p_labels, labels)
        else:
            loss_seg_dice = weights[1] * multiclass_dice_loss(p_labels, labels)
        # pixel cls
        loss_seg_bce = weights[2] * nn.BCEWithLogitsLoss(reduce=True)(logits, labels)
    else:
        loss_seg_dice, loss_seg_bce = torch.tensor(0).cuda(), torch.tensor(0).cuda()

    loss = loss_fc + loss_seg_bce + loss_seg_dice
    return loss, [loss_seg_bce, loss_seg_dice, loss_fc]

def combo_loss_onlypos(logits, fc=0, labels=0, labels_fc=0, weights=[0.1, 0, 1]):
    # weights -> [image_cls, pixel_seg, pixel_cls]
    # image class
    n_pos = labels_fc.sum()
    pos_idx = (labels_fc > 0.5)
    neg_idx = (labels_fc < 0.5)
    if weights[0]:
        loss_fc = weights[0] * nn.BCEWithLogitsLoss(reduce=True)(fc[neg_idx], labels_fc[neg_idx])
    else:
        loss_fc = torch.tensor(0).cuda()
    if weights[1] or weights[2]:
        # pixel seg
        if n_pos == 0:
            loss_seg_dice = torch.tensor(0).cuda()
        else:
            loss_seg_dice = weights[1] * SoftDiceLoss()(logits[pos_idx], labels[pos_idx])
        # pixel cls
        loss_seg_bce = weights[2] * nn.BCEWithLogitsLoss(reduce=True)(logits[pos_idx], labels[pos_idx])
    else:
        loss_seg_dice, loss_seg_bce = torch.tensor(0).cuda(), torch.tensor(0).cuda()

    loss = loss_fc + loss_seg_bce + loss_seg_dice
    return loss, [loss_seg_bce, loss_seg_dice, loss_fc]

def combo_loss_posDice(logits, fc=0, labels=0, labels_fc=0, weights=[0.1, 0, 1]):
    # weights -> [image_cls, pixel_seg, pixel_cls]
    # image class
    n_pos = labels_fc.sum()
    pos_idx = (labels_fc > 0.5)
    neg_idx = (labels_fc < 0.5)
    if weights[0]:
        loss_fc = weights[0] * nn.BCEWithLogitsLoss(reduce=True)(fc, labels_fc)
    else:
        loss_fc = torch.tensor(0).cuda()
    if weights[1] or weights[2]:
        # pixel seg
        if n_pos == 0:
            loss_seg_dice = torch.tensor(0).cuda()
        else:
            loss_seg_dice = weights[1] * SoftDiceLoss()(logits[pos_idx], labels[pos_idx])
        # pixel cls
        loss_seg_bce = weights[2] * nn.BCEWithLogitsLoss(reduce=True)(logits, labels)
    else:
        loss_seg_dice, loss_seg_bce = torch.tensor(0).cuda(), torch.tensor(0).cuda()

    loss = loss_fc + loss_seg_bce + loss_seg_dice
    return loss, [loss_seg_bce, loss_seg_dice, loss_fc]
