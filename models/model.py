# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:34:09



import segmentation_models_pytorch as smp
from torch import nn
import torch.nn.functional as F

from models.modelzoo import *
from models.utils import *
from models.Aspp import *
from models.Jpu import *

#smp.Unet(model_name, classes=args.n_classes, encoder_weights='imagenet', activation=None)
class model_cloud_smp(nn.Module):
    def __init__(self, framework, model_name, classes=1, pretrained=False, actication=None):
        super(model_cloud_smp, self).__init__()

        self.model_name = model_name
        self.framework = framework
        #self.model = Unet_plain(model_name, center, pretrained, vanilla)
        if pretrained:
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None
        if self.framework == 'Unet':
            self.model = smp.Unet(model_name, classes=classes, encoder_weights=encoder_weights, activation=actication)
        elif self.framework == 'FPN':
            self.model = smp.FPN(model_name, classes=classes, encoder_weights=encoder_weights, activation=actication)
        else:
            raise RuntimeError('%s Framework not implemented.' % framework)

        encoder_channels = self.model.encoder.out_shapes
        out_planes = encoder_channels[0]

        self.feature = nn.Conv2d(out_planes, 64, kernel_size=1)
        self.logit = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x):
        global_features = self.model.encoder(x)

        # class branch
        cls_feature = global_features[0]
        cls_feature = F.dropout(cls_feature, 0.5, training = self.model.training)
        cls_feature = F.adaptive_avg_pool2d(cls_feature, 1)
        cls_feature = self.feature(cls_feature)
        cls_feature = self.logit(cls_feature)
        # segmentation branch
        seg_feature = self.model.decoder(global_features)
        return seg_feature, cls_feature

class model_cloud_JPU(nn.Module):
    def __init__(self, model_name, classes=1, encoder_weights=True, activation=None):
        super(model_cloud_JPU, self).__init__()

        self.model_name = model_name
        if model_name == 'resnet34':
            if encoder_weights:
                self.encoder = resnet34(pretrained=True)
            else:
                self.encoder = resnet34(pretrained=False)
            self.planes = [512, 256, 128]
        elif model_name == 'resnet50':
            if encoder_weights:
                self.encoder = resnet50(pretrained=True)
            else:
                self.encoder = resnet50(pretrained=False)
            self.planes = [2048, 1024, 512]
        else:
            raise RuntimeError('%s not implemented.' % model_name)

        self.jpu = JPU(self.planes, 128)
        self.aspp = ASPP(512, 128, dilations=[1, 6, 12, 18], dropout=0.1)
        self.up = nn.UpsamplingNearest2d(scale_factor=8)
        self.logit = nn.Conv2d(128, classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.jpu(x[-3:][::-1])
        x = self.aspp(x)
        x = self.up(x)
        logit = self.logit(x)
        return logit
