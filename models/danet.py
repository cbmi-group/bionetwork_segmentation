###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, normalize
import sys

sys.path.append('/data/ldap_shared/home/s_lyr/code/er-network-segmentation/models/DANet')
from attention import PAM_Module
from attention import CAM_Module
from base import BaseNet



__all__ = ['DANet']


class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated U_DGF_Net network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in U_DGF_Net network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = interpolate(x[0], imsize, **self._up_kwargs)
        x[1] = interpolate(x[1], imsize, **self._up_kwargs)
        x[2] = interpolate(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)  # [output, pam_output, cam_output]


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


# def get_danet(dataset='pascal_voc', U_DGF_Net='resnet50', pretrained=False,
#             root='./pretrain_models', **kwargs):
#     r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
#     <https://arxiv.org/abs/1809.02983.pdf>`
#     """
#     acronyms = {
#         'pascal_voc': 'voc',
#         'pascal_aug': 'voc',
#         'pcontext': 'pcontext',
#         'ade20k': 'ade',
#         'cityscapes': 'cityscapes',
#     }
#     # infer number of classes
#     from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
#     model = DANet(datasets[dataset.lower()].NUM_CLASS, U_DGF_Net=U_DGF_Net, root=root, **kwargs)
#     if pretrained:
#         from .model_store import get_model_file
#         model.load_state_dict(torch.load(
#             get_model_file('fcn_%s_%s'%(U_DGF_Net, acronyms[dataset]), root=root)),
#             strict=False)
#     return model

if __name__ == '__main__':
    x = torch.ones([2, 1, 256, 256])
    model = DANet(2, backbone='resnet50')
    output = model(x)
    print(output[0].size(), output[1].size(), output[2].size())

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())







