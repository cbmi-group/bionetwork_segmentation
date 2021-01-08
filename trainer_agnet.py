# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import sys
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
from visdom import Visdom
import torch.nn.functional as F

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "datasets"))
sys.path.append(os.path.join(root_dir, "models/AGNet/code"))

from datasets.dataset import er_data_loader
from models.utils import init_weights
from models.agnet import AG_Net as agnet
from models.optimize import create_criterion, create_optimizer, update_learning_rate
from datasets.metric import *

# assert viz.check_connection()
viz = Visdom(env="test", port=4321)
# assert viz.check_connection()

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def print_table(data):
    col_width = [max(len(item) for item in col) for col in data]
    for row_idx in range(len(data[0])):
        for col_idx, col in enumerate(data):
            item = col[row_idx]
            align = '<' if not col_idx == 0 else '>'
            print(('{:' + align + str(col_width[col_idx]) + '}').format(item), end=" ")
        print()

EPS = 1e-12

def train_one_epoch(epoch, total_steps, dataloader, model,
                    device, criterion, optimizer, lr, lr_decay,
                    display_iter, log_file, show):
    model.train()

    smooth_loss = 0.0
    current_step = 0
    t0 = 0.0


    for inputs in dataloader:
        t1 = time.time()

        images = inputs['image'].to(device)
        labels = inputs['mask'].to(device)
        labels = labels.long().squeeze()

        # forward pass
        out, side_5, side_6, side_7, side_8 = model(images)
        out = torch.log(F.softmax(out, dim=1) + EPS)
        loss = criterion(out, labels)
        loss += criterion(torch.log(F.softmax(side_5, dim=1) + EPS), labels)
        loss += criterion(torch.log(F.softmax(side_6, dim=1) + EPS), labels)
        loss += criterion(torch.log(F.softmax(side_7, dim=1) + EPS), labels)
        loss += criterion(torch.log(F.softmax(side_8, dim=1) + EPS), labels)
        out = torch.log(F.softmax(side_8) + EPS)
        seg = torch.argmax(out, dim=1)

        t0 += (time.time() - t1)

        total_steps += 1
        current_step += 1
        smooth_loss += loss.item()

        # backpropagate when training
        optimizer.zero_grad()
        lr_update = update_learning_rate(optimizer, epoch, lr, step=lr_decay)
        loss.backward()
        optimizer.step()


        if show:
            raw_img = images[0, 0, :, :].data.cpu().numpy()
            raw_mask = labels[0, :, :].data.cpu().numpy() * 255
            raw_mask = raw_mask.astype(np.uint8)
            prd_score = seg[0, :, :].data.cpu().numpy()
            prd_mask = prd_score > 0.5

            viz.heatmap(raw_img, win="train_image", opts={"title": "Train Image", "colormap": "Gray"})
            viz.image(raw_mask[::-1, :], win="train_mask", opts={"title": "Train Mask"})
            viz.image(prd_mask[::-1, :].astype(np.uint8) * 255, win="pred_mask", opts={"title": "Predicted Mask"})

        # log loss
        if total_steps % display_iter == 0:
            smooth_loss = smooth_loss / current_step
            message = "Epoch: %d Step: %d LR: %.6f Loss: %.4f Runtime: %.2fs/%diters." % (
            epoch + 1, total_steps, lr_update, smooth_loss, t0, display_iter)
            print("==> %s" % (message))
            with open(log_file, "a+") as fid:
                fid.write('%s\n' % message)

            # visualize training loss
            viz.line(Y=[smooth_loss], X=torch.Tensor([total_steps]), win='train loss', update='append',
                     opts=dict(title="Training Loss", xlabel="Step", ylabel="Smoothed Loss"))

            t0 = 0.0
            current_step = 0
            smooth_loss = 0.0

    return total_steps


def eval_one_epoch(epoch, dataloader, model, device, train_dir, log_file):
    with torch.no_grad():
        model.eval()

        total_iou = 0.0
        total_img = 0
        total_f1 = 0.0
        total_distance = 0.0
        total_acc = 0.0


        for inputs in dataloader:
            images = inputs['image'].to(device)
            labels = inputs['mask']

            total_img += len(images)

            out, side_5, side_6, side_7, side_8 = model(images)
            outputs = torch.log(F.softmax(side_8, dim=1) + 1e-10)
            seg = torch.argmax(outputs, dim=1).unsqueeze(dim=1)
            preds = seg.cpu()

            # metric
            val_acc = acc(preds, labels)
            total_acc += val_acc

            val_iou = IoU(preds, labels)
            total_iou += val_iou

            h_distance = hausdorff_distance(preds, labels)
            total_distance += h_distance

            val_f1 = F1_score(preds, labels)
            total_f1 += val_f1

        # iou
        epoch_iou = total_iou / total_img
        viz.line(Y=torch.Tensor([epoch_iou]),
                 X=torch.Tensor([epoch + 1]),
                 win='val IoU',
                 update='append',
                 opts=dict(title="Validation IoU", xlabel="Epoch", ylabel="IoU"))

        epoch_f1 = total_f1 / total_img
        viz.line(Y=torch.Tensor([epoch_f1]),
                 X=torch.Tensor([epoch + 1]),
                 win='val F1',
                 update='append',
                 opts=dict(title="Validation F1", xlabel="Epoch", ylabel="F1"))

        epoch_distance = total_distance / total_img
        viz.line(Y=torch.Tensor([epoch_distance]),
                 X=torch.Tensor([epoch + 1]),
                 win='val hausdorff distance',
                 update='append',
                 opts=dict(title="Validation h_distance", xlabel="Epoch", ylabel="h_distance"))

        epoch_acc = total_acc / total_img
        message = "total Threshold: {:.2f} =====> Evaluation IOU: {:.4f}; F1_score: {:.4f}; Haussdorf distance:{:.2f}".format(
                0.5, epoch_iou, epoch_f1, epoch_distance)
        print("==> %s" % (message))
        with open(log_file, "a+") as fid:
            fid.write('%s\n' % message)


    return epoch_acc, epoch_iou, epoch_f1, epoch_distance


def train_eval_model(opts):
    # parse model configuration
    num_epochs = opts["num_epochs"]
    train_batch_size = opts["train_batch_size"]
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']

    opti_mode = opts["optimizer"]
    loss_criterion = opts["loss_criterion"]
    lr = opts["lr"]
    lr_decay = opts["lr_decay"]
    wd = opts["weight_decay"]

    gpus = opts["gpu_list"].split(',')
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]
    train_dir = opts["log_dir"]

    train_data_dir = opts["train_data_dir"]
    eval_data_dir = opts["eval_data_dir"]

    pretrained = opts["pretrained_model"]
    resume = opts["resume"]
    display_iter = opts["display_iter"]
    save_epoch = opts["save_every_epoch"]
    show = opts["vis"]

    # backup train configs
    log_file = os.path.join(train_dir, "log_file.txt")
    os.makedirs(train_dir, exist_ok=True)
    model_dir = os.path.join(train_dir, "code_backup")
    os.makedirs(model_dir, exist_ok=True)
    if resume is None and os.path.exists(log_file): os.remove(log_file)
    shutil.copy("./models/agnet.py", os.path.join(model_dir, "agnet.py"))
    shutil.copy("./trainer_agnet.py", os.path.join(model_dir, "trainer_agnet.py"))
    shutil.copy("./datasets/dataset.py", os.path.join(model_dir, "dataset.py"))

    ckt_dir = os.path.join(train_dir, "checkpoints")
    os.makedirs(ckt_dir, exist_ok=True)

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    n = 0
    for key, value in opts.items():
        table_key.append(key)
        table_value.append(str(value))
        n += 1
    print_table([table_key, ["="] * n, table_value])

    # with open(log_file, "a+") as fid:
    #     for key, value in opts.items():
    #         fid.write(key + ": " + str(value))
    # print("*" * 50)

    # format gpu list
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)

    # dataloader
    print("==> Create dataloader")
    dataloaders_dict = {"train": er_data_loader(train_data_dir, train_batch_size, dataset_type, is_train=True),
                        "eval": er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)}

    # define parameters of two networks
    print("==> Create network")
    num_channels = 1
    num_classes = 2
    unet_model = agnet(num_classes)
    init_weights(unet_model)

    # loss layer
    criterion = loss_criterion

    best_acc = 0.0
    start_epoch = 0

    # load pretrained model
    if pretrained is not None and os.path.isfile(pretrained):
        print("==> Train from model '{}'".format(pretrained))
        checkpoint_gan = torch.load(pretrained)
        unet_model.load_state_dict(checkpoint_gan['model_state_dict'])
        print("==> Loaded checkpoint '{}')".format(pretrained))
        for param in unet_model.parameters():
            param.requires_grad = False

    # resume training
    elif resume is not None and os.path.isfile(resume):
        print("==> Resume from checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model_dict = unet_model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        unet_model.load_state_dict(pretrained_dict)
        print("==> Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch'] + 1))

    # train from scratch
    else:
        print("==> Train from initial or random state.")

    # define mutiple-gpu mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_model.cuda()
    unet_model = nn.DataParallel(unet_model)

    # print learnable parameters
    print("==> List learnable parameters")
    for name, param in unet_model.named_parameters():
        if param.requires_grad == True:
            print("\t{}".format(name))
    params_to_update = [{'params': unet_model.parameters()}]

    # define optimizer
    print("==> Create optimizer")
    optimizer = create_optimizer(params_to_update, opti_mode, lr=lr, momentum=0.9, wd=wd)
    if resume is not None and os.path.isfile(resume): optimizer.load_state_dict(checkpoint['optimizer'])

    # start training
    since = time.time()

    # Each epoch has a training and validation phase
    print("==> Start training")
    total_steps = 0

    val_iou = []

    for epoch in range(start_epoch, num_epochs):

        print('-' * 50)
        print("==> Epoch {}/{}".format(epoch + 1, num_epochs))

        total_steps = train_one_epoch(epoch, total_steps,
                                      dataloaders_dict['train'],
                                      unet_model, device,
                                      criterion, optimizer, lr, lr_decay,
                                      display_iter, log_file, show)

        epoch_acc, epoch_iou, epoch_f1, epoch_distance = eval_one_epoch(epoch, dataloaders_dict['eval'], unet_model, device, train_dir, log_file)

        val_iou.append('{:4f}'.format(epoch_iou.item()))
        #plot_val(val_iou, train_dir)

        if best_acc < epoch_acc and epoch >= 3:
            best_acc = epoch_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': unet_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(ckt_dir, "best.pth"))

        if (epoch + 1) % save_epoch == 0 and (epoch + 1) >= 20:
            torch.save({'epoch': epoch,
                        'model_state_dict': unet_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': epoch_acc,
                        'best_f1': epoch_f1,
                        'best_haussdorf_distance': epoch_distance,
                        'best_iou': epoch_iou},
                       os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))

    time_elapsed = time.time() - since
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('==> Best val Acc: {:4f}'.format(best_acc))




if __name__ == '__main__':
    # parse model configuration
    opts = dict()
    opts['dataset_type'] = 'retina'

    opts["num_epochs"] = 30
    opts["train_data_dir"] = "./datasets/train_er_aug_v1.txt"
    opts["eval_data_dir"] = "./datasets/test_er.txt"
    opts["train_batch_size"] = 16
    opts["eval_batch_size"] = 32
    opts["optimizer"] = "SGD"
    opts["loss_criterion"] = nn.NLLLoss2d()
    opts["lr"] = 0.01
    opts["lr_decay"] = 5
    opts["weight_decay"] = 0.0005
    opts["gpu_list"] = "0,1,2,3"
    opts["log_dir"] = "./train_log/stare_agnet_20200805"
    opts["pretrained_model"] = None
    opts["resume"] = None
    opts["display_iter"] = 10
    opts["save_every_epoch"] = 2
    opts["vis"] = True

    train_eval_model(opts)
