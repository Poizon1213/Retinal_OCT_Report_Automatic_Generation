import torch.nn as nn
import torch
import torch.nn.functional as F
from models.base_models.resnet import resnet34, resnet50, resnet101
from models.base_models.resnest import resnest50, resnest101, resnest200, resnest269
from models.base_models.PyConvResNet.pyconvresnet import pyconvresnet50, pyconvresnet101
from models.base_models.EfficientNet.model import EfficientNet
from models.base_models.resnext import resnext50_32x4d
from models.base_models.densenet import densenet121, densenet169, densenet201
from models.base_models.sedensenet import se_densenet121
# from model.base_models.densenet_changemodule import densenet121, densenet169, densenet201

class classifer_base(nn.Module):

    def __init__(self, backbone='resnet50', pretrained_base=True, n_class=2, in_channel=3, **kwargs):
        super(classifer_base, self).__init__()
        self.in_channel = in_channel
        self.net = backbone
        if pretrained_base:
            self.n_class = 1000
        else:
            self.n_class = n_class

        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.backbone.conv1 = nn.Conv2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained_base, num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.backbone.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest50':
            self.backbone = resnest50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest101':
            self.backbone = resnest101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest200':
            self.backbone = resnest200(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnest269':
            self.backbone = resnest269(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet50':
            self.backbone = pyconvresnet50(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'pyconvresnet101':
            self.backbone = pyconvresnet101(pretrained=pretrained_base,  num_classes=self.n_class, in_channel=in_channel, **kwargs)
            self.base_channel = [256, 512, 1024, 2048]
        elif "efficientnet" in backbone:
            if pretrained_base:
                self.backbone = EfficientNet.from_pretrained(backbone)
            else:
                self.backbone = EfficientNet.from_name(backbone, in_channels=in_channel, num_classes=n_class)
            self.base_channel = [self.backbone.out_channels]
        elif backbone == 'densenet121':
            self.backbone = densenet121(pretrained=pretrained_base, num_classes=self.n_class)
            # self.backbone.features.conv0 = nn.Conv2d(128, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'sedensenet121':
            self.backbone = se_densenet121(pretrained=pretrained_base, num_classes=self.n_class)
            self.backbone.features.conv0 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet169':
            self.backbone = densenet169(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'densenet201':
            self.backbone = densenet201(pretrained=pretrained_base, num_classes=self.n_class)
            self.base_channel = [self.backbone.num_features]
        elif backbone == 'resnext50':
            self.backbone = resnext50_32x4d(pretrained=pretrained_base)
            self.base_channel = [256, 512, 1024, 2048]
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gdc = nn.Conv2d(1024, 1024, (7, 7))

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.base_channel[-1], n_class)
        # self.fc1 = nn.Linear(1024, n_class)
        # # 冻结参数
        # for p in self.backbone.parameters():
        #     p.requires_grad = False


    def forward(self, x1, x2):
        x1 = self.backbone.extract_features(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.drop(x1)

        x2 = self.backbone.extract_features(x2)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.drop(x2)

        x = torch.cat([x1, x2], dim=1)
        # x = self.fc1(x)

        return x
        # x = self.backbone.extract_features(x)
        # x = self.avgpool(x)
        #
        # x = x.view(x.size(0), -1)
        # x = self.drop(x)
        # x = self.fc1(x)
        #
        # return x