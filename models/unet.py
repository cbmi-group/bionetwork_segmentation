import torch.nn.functional as F
import torch
import torch.nn as nn
import glob
import os
import numpy as np
import cv2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



'''
model
'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, n_classes)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        # decoder
        o_4 = self.up1(x5, x4)
        o_3 = self.up2(o_4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)


        if self.n_classes > 1:
            seg = F.softmax(o_seg, dim=1)
            return seg
        elif self.n_classes == 1 :
            seg = torch.sigmoid(o_seg)
            return seg


class UNet_multiloss(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(UNet_multiloss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, n_classes)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        # decoder
        o_4 = self.up1(x5, x4)
        o_3 = self.up2(o_4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)


        if self.n_classes > 1:
            seg = F.softmax(o_seg, dim=1)
            return seg
        elif self.n_classes == 1 :
            seg = torch.sigmoid(o_seg)
            return seg

    def sort_prd(self, feature, prd, mask, ratio):
        b,c,h,w = prd.size()
        forward_len = torch.sum(mask, dim=(1, 2, 3))
        backward_len = torch.sum(1-mask, dim=(1,2,3))
        forward_cut_indice = torch.ceil(forward_len * ratio)
        backward_cut_indice = torch.ceil(backward_len * ratio)

        forward_prd = prd * mask
        forward_prd = forward_prd.view(b,c,-1).transpose(0,2,1)
        sort_forward_prd, forward_prd_indice = torch.sort(forward_prd, dim=-1, descending=True)
        low_forward_indice = forward_prd_indice[:forward_cut_indice]
        high_forward_indice = forward_prd_indice[forward_cut_indice:forward_len]

        forward_feature = feature.view(b,c,-1).transpose(0,2,1)
        low_forward_feature = torch.gather(forward_feature, -1, low_forward_indice)
        high_forward_feature = torch.gather(forward_feature, -1, high_forward_indice)



        bacward_prd = prd * (1 - mask)
        bacward_prd = bacward_prd.view(b,c,-1).transpose(0,2,1)
        sort_backward_prd, backward_prd_indice = torch.sort(bacward_prd, dim=-1, descending=True)
        low_backward_indice = backward_prd_indice[:backward_cut_indice]
        high_backward_indice = backward_prd_indice[backward_cut_indice:backward_len]

        backward_feature = feature.view(b,c,-1).transpose(0,2,1)
        low_backward_feature = torch.gather(backward_feature, -1, low_backward_indice)
        high_backward_feature = torch.gather(backward_feature, -1, high_backward_indice)

        return (low_forward_feature, high_forward_feature), (low_backward_feature, high_backward_feature)








if __name__ == '__main__':

    def sort_prd(feature, prd, mask, ratio):
        b,c,h,w = prd.size()
        forward_len = torch.sum(mask, dim=(1, 2, 3))
        backward_len = torch.sum(1-mask, dim=(1,2,3))
        forward_cut_indice = torch.ceil(forward_len * ratio).long()
        backward_cut_indice = torch.ceil(backward_len * ratio)

        forward_prd = prd * mask
        forward_prd = forward_prd.view(b,c,-1).transpose(2,1)
        sort_forward_prd, forward_prd_indice = torch.sort(forward_prd, dim=1, descending=True)

        forward_feature = feature.view(b, feature.size()[1], -1).transpose(2, 1)

        # low_forward_indice = torch.zeros_like()
        for i in range(b):
            low_forward_indice_i = forward_prd_indice[i][:forward_cut_indice[i], :]
            high_forward_indice_i = forward_prd_indice[i][forward_cut_indice[i]:forward_len[i].long(),:]

            low_forward_feature = torch.gather(forward_feature[i], -1, low_forward_indice_i)
            high_forward_feature = torch.gather(forward_feature[i], -1, high_forward_indice_i)



        bacward_prd = prd * (1 - mask)
        bacward_prd = bacward_prd.view(b,c,-1).transpose(2,1)
        sort_backward_prd, backward_prd_indice = torch.sort(bacward_prd, dim=-1, descending=True)
        low_backward_indice = backward_prd_indice[:backward_cut_indice]
        high_backward_indice = backward_prd_indice[backward_cut_indice:backward_len]

        backward_feature = feature.view(b,c,-1).transpose(2,1)
        low_backward_feature = torch.gather(backward_feature, -1, low_backward_indice)
        high_backward_feature = torch.gather(backward_feature, -1, high_backward_indice)

        return (low_forward_feature, high_forward_feature), (low_backward_feature, high_backward_feature)

    feature = torch.randn([2,2,3,3])
    prd = torch.randn([2,1,3,3])
    mask = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2,1,1,1)
    print(1-mask)
    a,b = sort_prd(feature, prd, mask, 0.5)
    print(a)

    # a = torch.gather(feature, -1, index=(torch.ones(2,2,3,3).long()))
    # print(a)




