# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:51



from models.Aspp import *
from models.modelzoo import *
from models.utils import *
import efficientnet_pytorch

class seresnext(nn.Module):
    def __init__(self, net_name, num_classes,
                 pretrained=True):
        super().__init__()
        if pretrained:
            pretrained = 'imagenet'
        if net_name == 'seresnext50':
            self.net = se_resnext50_32x4d(pretrained=pretrained)
            self.planes = 2048
        else:
            RuntimeError('%s not implemented.'%(net_name))

        self.feature = nn.Conv2d(self.planes, 64, kernel_size=1)
        self.logit = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x):
        x = self.net(x)[-1]
        # print(x.size())
        #x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        x = self.logit(x)
        return x


class resnet(nn.Module):
    def __init__(self, net_name, num_classes,
                 pretrained=True):
        super().__init__()
        if net_name == 'resnet50':
            self.net = resnet50(pretrained=pretrained)
            self.planes = 2048
        elif net_name == 'resnet34':
            self.net = resnet34(pretrained=pretrained)
            self.planes = 512
        else:
            RuntimeError('%s not implemented.'%(net_name))

        self.feature = nn.Conv2d(self.planes, 64, kernel_size=1)
        self.logit = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x):
        x = self.net(x)[-1]
        x = F.dropout(x, 0.5, training=self.training)
        #print(x.size())
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        x = self.logit(x)
        return x

class efficientnet(nn.Module):
    def __init__(self, net_name, num_classes,
                 pretrained=True):
        super().__init__()
        if pretrained:
            self.enet = efficientnet_pytorch.EfficientNet.from_pretrained(net_name, num_classes)
        else:
            self.enet = efficientnet_pytorch.EfficientNet.from_name(net_name, {'num_classes': num_classes})

        self._in_features = self.enet._fc.in_features
        self.feature = nn.Conv2d(self._in_features, 64, kernel_size=1)
        self.logit = nn.Conv2d(64, num_classes, kernel_size=1)
        #self.logit = nn.Linear(self._in_features, num_classes)

    def forward(self, x):
        x = self.enet.extract_features(x)
        # print(x.size())
        #x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.enet._dropout:
            x = F.dropout(x, self.enet._dropout, self.enet.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        x = self.logit(x)
        return x
