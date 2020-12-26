# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import torch

from datasets.dataset import er_data_loader
from models.unet import UNet as u_net
from models.nested_unet import NestedUNet as u_net_plus
from models.deeplab_v3 import DeepLab as deeplab
from models.pe_net import PE_Net
from models.agnet import AG_Net as ag_net

import cv2
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)


'''
evaluation
'''
def eval_model(opts):
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    load_epoch = opts['load_epoch']
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    eval_data_dir = opts["eval_data_dir"]
    dataset_name = os.path.split(eval_data_dir)[-1].split('.')[0]

    train_dir = opts["train_dir"]
    model_type = opts['model_type']

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]), 'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    viz_dir = train_dir.replace('checkpoints', 'viz')
    seg_save = os.path.join(viz_dir, 'seg')
    p_seg_save = os.path.join(viz_dir, 'p_seg')
    if not os.path.exists(seg_save): os.makedirs(seg_save)
    if not os.path.exists(p_seg_save): os.makedirs(p_seg_save)

    # dataloader
    print("==> Create dataloader")
    dataloader= er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train = False)

    # define network
    print("==> Create network")

    model = None

    if model_type == 'unet':
        model = u_net(1,1)
    elif model_type == 'unetPlus':
        model = u_net_plus(1,1)
    elif model_type == 'agnet':
        model = ag_net(2)
    elif model_type == 'deeplab':
        model = deeplab(backbone='resnet50', output_stride=16)
    elif model_type == 'penet':
        model = PE_Net()

    # load trained model
    pretrain_model = os.path.join(train_dir, "checkpoints_" + str(load_epoch) + ".pth")

    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)

        model.load_state_dict(c_checkpoint["model_state_dict"])

        print("==> Loaded pretrianed model checkpoint '{}'.".format(pretrain_model))

    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_list)


    # enable evaluation mode
    with torch.no_grad():
        model.eval()

        total_img = 0


        for inputs in dataloader:
            images = inputs["image"].cuda()

            img_name = inputs['ID']

            total_img += len(images)

            p_seg = 0
            # unet
            if model_type == 'unet':
                p_seg = model(images)

            elif model_type == 'unetPlus':
                p_seg = model(images)
                p_seg = p_seg[-1]

            elif model_type == 'penet':
                p_seg, p_seg_down = model(images)

            # agnet
            elif model_type == 'agnet':
                out, side_5, side_6, side_7, side_8 = model(images)
                p_seg = F.softmax(side_8, dim=1)

            # deeplab
            elif model_type == 'deeplab':
                p_seg = model(images)

            for i in range(len(images)):

                print('predict image: {}'.format(img_name[i]))

                if model_type == 'agnet':
                    np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), p_seg[i][1].cpu().numpy().astype(np.float32))
                    cv2.imwrite(os.path.join(model_score_dir, img_name[i]), p_seg[i][1].cpu().numpy().astype(np.float32))
                else:
                    np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), p_seg[i][0].cpu().numpy().astype(np.float32))
                    cv2.imwrite(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.tif'), p_seg[i][0].cpu().numpy().astype(np.float32))

        print("validation image number {}".format(total_img))


if __name__ == "__main__":
    opts = dict()
    opts['dataset_type'] = 'er'

    opts["eval_batch_size"] = 32
    opts["gpu_list"] = "0,1,2,3"
    opts["train_dir"] = "./train_log/er_train_aug_v1_20200727_bceloss/checkpoints"
    opts["eval_data_dir"] = "./datasets/test_er.txt"

    # model_type = [unet, unetPlus, agnet, deeplab, penet]
    opts['model_type'] = 'unetPlus'
    opts["load_epoch"] = 30

    eval_model(opts)


