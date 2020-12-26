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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



'''
model
'''
class PE_Net(nn.Module):
    def __init__(self, in_dim=1, base_dim=32, n_classes=1, bilinear=False):
        super(PE_Net, self).__init__()
        self.n_channels = in_dim
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.layer_branch_up_1 = DoubleConv(in_dim, base_dim)
        self.layer_branch_up_2 = DoubleConv(base_dim, base_dim * 2)
        self.layer_branch_up_3 = DoubleConv(base_dim * 2, base_dim * 4)

        self.down = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=2, padding=1)

        self.layer_branch_up_4 = ConvLayer(base_dim * 8, base_dim * 4)
        self.layer_branch_up_5 = ConvLayer(base_dim * 4, base_dim * 2)
        self.layer_branch_up_6 = ConvLayer(base_dim * 2, base_dim)

        self.downsample_x = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.input_down = nn.MaxPool2d(2)
        self.layer_branch_down_1 = ConvLayer(in_dim, base_dim)
        self.layer_branch_down_2 = DoubleConv(base_dim * 2, base_dim * 2)
        self.layer_branch_down_3 = DoubleConv(base_dim * 2, base_dim * 4)

        self.up = nn.ConvTranspose2d(base_dim * 4, base_dim * 4, kernel_size=2, stride=2)

        self.layer_branch_down_4 = ConvLayer(base_dim * 4, base_dim)

        self.out_branch_up = OutConv(base_dim, n_classes)
        self.out_branch_down = OutConv(base_dim, n_classes)

    def forward(self, x):
        x_up_1 = self.layer_branch_up_1(x)
        x_up_2 = self.layer_branch_up_2(x_up_1)
        x_up_3 = self.layer_branch_up_3(x_up_2)

        x_down_1 = self.layer_branch_down_1(self.downsample_x(x))
        x_up_1_down = self.input_down(x_up_1)
        x_down_2 = self.layer_branch_down_2(torch.cat((x_down_1, x_up_1_down), dim=1))
        x_down_3 = self.layer_branch_down_3(x_down_2)

        x_down_3_up = self.up(x_down_3)

        x_up_4 = self.layer_branch_up_4(torch.cat((x_up_3, x_down_3_up), dim=1))
        x_up_5 = self.layer_branch_up_5(x_up_4)
        x_up_6 = self.layer_branch_up_6(x_up_5)

        x_down_4 = self.layer_branch_down_4(x_down_3)

        pred_branch_up = torch.sigmoid(self.out_branch_up(x_up_6))
        pred_branch_down = torch.sigmoid(self.out_branch_down(x_down_4))

        return pred_branch_up, pred_branch_down


class ER_Net_V1(nn.Module):
    def __init__(self, in_dim=1, base_dim=32, n_classes=1, bilinear=False):
        super(ER_Net_V1, self).__init__()
        self.n_channels = in_dim
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inlayer_branch_up = nn.Sequential(nn.Conv2d(in_dim, base_dim, 3, 1, 1),
                                               nn.BatchNorm2d(base_dim),
                                               nn.ReLU(inplace=True))
        self.layer_branch_up_1 = DoubleConv(base_dim, base_dim*2)
        self.layer_branch_up_2 = DoubleConv(base_dim*2, base_dim*4)
        self.layer_branch_up_3 = DoubleConv(base_dim*4, base_dim*8)

        # downsampling branch
        self.downsample_x = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.inlayer_branch_down = nn.Sequential(nn.Conv2d(in_dim, base_dim, 3, 1, 1),
                                                 nn.BatchNorm2d(base_dim),
                                                 nn.ReLU(inplace=True))
        self.down = Down(base_dim*2, base_dim)
        self.layer_branch_down_1 = DoubleConv(base_dim*2, base_dim*4)
        self.layer_branch_down_2 = DoubleConv(base_dim*4, base_dim*8)
        self.layer_branch_down_3 = DoubleConv(base_dim*8, base_dim*16)
        self.layer_branch_down_4 = DoubleConv(base_dim*16, base_dim)

        # upsampling
        self.up = Up(base_dim*16, base_dim*8, bilinear)
        self.layer_branch_up_4 = DoubleConv(base_dim*8, base_dim*4)
        self.layer_branch_up_5 = DoubleConv(base_dim*4, base_dim*2)
        self.layer_branch_up_6 = DoubleConv(base_dim*2, base_dim)

        self.out_branch_up = OutConv(base_dim, n_classes)
        self.out_branch_down = OutConv(base_dim, n_classes)

    def forward(self, x):

        x_in_up = self.inlayer_branch_up(x)
        x_in_down = self.inlayer_branch_down(self.downsample_x(x))

        x_up_1 = self.layer_branch_up_1(x_in_up)
        x_up_2 = self.layer_branch_up_2(x_up_1)
        x_up_3 = self.layer_branch_up_3(x_up_2)

        x_up_down = self.down(x_up_1)
        x_down_1 = self.layer_branch_down_1(torch.cat((x_in_down, x_up_down), dim=1))
        x_down_2 = self.layer_branch_down_2(x_down_1)
        x_down_3 = self.layer_branch_down_3(x_down_2)
        x_down_4 = self.layer_branch_down_4(x_down_3)

        x_down_up = self.up(x_down_3, x_up_3)

        x_up_4 = self.layer_branch_up_4(x_down_up)
        x_up_5 = self.layer_branch_up_5(x_up_4)
        x_up_6 = self.layer_branch_up_6(x_up_5)

        out_up = self.out_branch_up(x_up_6)
        out_down = self.out_branch_down(x_down_4)

        pred_branch_up = torch.sigmoid(out_up)
        pred_branch_down = torch.sigmoid(out_down)

        return pred_branch_up, pred_branch_down





