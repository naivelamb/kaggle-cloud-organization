# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:34:14



from common import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class UpBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            SEModule(in_c, reduction=16),
            nn.Conv2d(in_c, mid_c, kernel_size=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=1),
            nn.ReLU()
        )
        # todo add seblock!

    def forward(self, x):
        return self.conv(x)

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

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        eps = 1e-5
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = torch.sum(m1 * m2, 1)
        union = torch.sum(m1, dim=1) + torch.sum(m2, dim=1)
        score = (2*intersection + eps)/(union + eps)
        score = (1 - score).mean()
        return score

def multiclass_dice_loss(logits, targets):
    loss = 0
    dice = SoftDiceLoss()
    num_classes = targets.size(1)
    for class_nr in range(num_classes):
        loss += dice(logits[:, class_nr, :, :], targets[:, class_nr, :, :])
    return loss/num_classes

class SoftDiceLoss_binary(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        inter = torch.sum(input * target, 1) + smooth
        union = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth

        score = torch.sum(2.0 * inter / union) / float(batch_size)
        score = 1.0 - torch.clamp(score, 0.0, 1.0 - 1e-7)

        return score

if __name__ == '__main__':
    input = torch.rand((4, 1, 128))
    target = torch.ones((4, 1, 128))
    loss = FocalLoss(2, 0.2)(input, target)
    print(loss.item())
