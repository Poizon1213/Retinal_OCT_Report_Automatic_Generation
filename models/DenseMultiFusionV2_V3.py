import torch
import torch.nn as nn
import torch.nn.functional as F
from config import encoder_dim

class Multi_Scale_Module(nn.Module):
    def __init__(self):
        super(Multi_Scale_Module, self).__init__()

        self.att = nn.Sequential(
            nn.BatchNorm2d(1920),
            nn.ReLU(inplace=True),
            nn.Conv2d(1920,3,kernel_size=(1,1),bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        self.conv_1_4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14,14)),
            nn.Conv2d(128,128,kernel_size=(3,3),padding=1,bias=False),
        )
        self.conv_2_4 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14,14)),
            nn.Conv2d(256,256,kernel_size=(3,3),padding=1,bias=False),
        )
        self.conv_3_4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=(3,3),padding=1,bias=False),
        )


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,input_1,input_2,input_3,input_4):
        # print('input_1.shape: ', input_1.shape)
        # print('input_2.shape: ', input_2.shape)
        # print('input_3.shape: ', input_3.shape)
        # print('input_4.shape: ', input_4.shape)

        input_1_4 = self.conv_1_4(input_1)
        input_2_4 = self.conv_2_4(input_2)
        input_3_4 = self.conv_3_4(input_3)

        x_cat = torch.cat([input_1_4,input_2_4,input_3_4,input_4],dim=1)
        att = self.att(x_cat)


        out = torch.cat([
            torch.unsqueeze(att[:, 0], dim=1) * input_1_4,
            torch.unsqueeze(att[:, 1], dim=1) * input_2_4,
            torch.unsqueeze(att[:, 2], dim=1) * input_3_4,
            input_4
        ],dim=1)

        return out


class FeatureFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FeatureFusion, self).__init__()
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch,out_ch,kernel_size=(3,3),padding=1,bias=False),
        )
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,input):
        out = self.Conv(input)
        return out


class DenseMultiFusionV2_V3(nn.Module):
    def __init__(self):
        print("=========== DenseMultiFusionV2_V3 ===========")
        super(DenseMultiFusionV2_V3, self).__init__()
        from torchvision.models.densenet import densenet121
        densenet = densenet121(pretrained=True)
        # print(list(densenet.children()))
        module_Features = list(densenet.children())[:-1][0]
        # print(module_Features)
        module_list = list(module_Features.children())
        # print(module_list)
        # print(len(module_list))
        self.Conv_Layer = nn.Sequential(
            module_list[0],
            module_list[1],
            module_list[2],
            module_list[3]
        )
        self.Dense_L1 = nn.Sequential(
            module_list[4],
            module_list[5],#Trans
        )
        self.Dense_L2 = nn.Sequential(
            module_list[6],
            module_list[7],#Trans
        )
        self.Dense_L3 = nn.Sequential(
            module_list[8],
            module_list[9],
        )
        self.Dense_L4 = nn.Sequential(
            module_list[10],
        )
        self.Ms_Fusion = Multi_Scale_Module()

        self.Smoonth = nn.Sequential(
            nn.BatchNorm2d(1920),
            nn.ReLU(inplace=True),
            nn.Conv2d(1920,1024,kernel_size=(1,1),stride=(1,1),bias=False),
            module_list[11],
        )

        self.FF_1 = FeatureFusion(in_ch=128,out_ch=128)
        self.FF_2 = FeatureFusion(in_ch=256,out_ch=256)
        self.FF_3 = FeatureFusion(in_ch=512,out_ch=512)
        self.FF_4 = FeatureFusion(in_ch=1024,out_ch=1024)

        self.avgpool_fun = nn.AvgPool2d(14)  #
        self.dropout = nn.Dropout(0.5)
        self.affine_classifier = nn.Linear(1024, 11)
        self.sigmoid = nn.Sigmoid()


    def forward(self,input):
        x_conv_1 = self.Conv_Layer(input[0])
        x_l1_1 = self.Dense_L1(x_conv_1)
        x_l2_1 = self.Dense_L2(x_l1_1)
        x_l3_1 = self.Dense_L3(x_l2_1)
        x_l4_1 = self.Dense_L4(x_l3_1)

        x_conv_2 = self.Conv_Layer(input[1])
        x_l1_2 = self.Dense_L1(x_conv_2)
        x_l2_2 = self.Dense_L2(x_l1_2)
        x_l3_2 = self.Dense_L3(x_l2_2)
        x_l4_2 = self.Dense_L4(x_l3_2)

        FF1 = self.FF_1(x_l1_1+x_l1_2)
        FF2 = self.FF_2(x_l2_1+x_l2_2)
        FF3 = self.FF_3(x_l3_1+x_l3_2)
        FF4 = self.FF_4(x_l4_1+x_l4_2)
        x_top = self.Ms_Fusion(FF1,FF2,FF3,FF4)
        x_smooth = self.Smoonth(x_top)

        x_smooth = x_smooth.permute(0, 2, 3, 1)
        return x_smooth


if __name__ == '__main__':
    images_1 = torch.rand(4, 3, 448, 448).cuda(0)
    images_2 = torch.rand(4, 3, 448, 448).cuda(0)
    model = DenseMultiFusionV2_V3()
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.cuda(0)
    print(model((images_1,images_2))[1].size())

