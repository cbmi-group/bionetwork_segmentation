import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F

def create_criterion(criterion="crossentropy"):
    if criterion == "crossentropy":
        return nn.CrossEntropyLoss()
    elif criterion == "bce":
        return nn.BCELoss()
    elif criterion == "bcelog":
        return nn.BCEWithLogitsLoss()
    elif criterion == "L1":
        return nn.L1Loss()
    elif criterion == "MSE":
        return nn.MSELoss()
    elif criterion == "focal":
        return FocalLoss2d()
    elif criterion == "wbce":
        return weighted_edge_loss()
    elif criterion == "iou":
        return soft_iou_loss()
    elif criterion == 'dice':
        return DiceLoss()


def create_optimizer(parameters, mode="SGD", lr=0.001, momentum=0.9, wd=0.0005, beta1=0.5, beta2=0.999):
    if mode == "SGD":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)
    elif mode == "Adam":
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=wd)


def update_learning_rate(optimizer, epoch, lr, step=30, gamma=0.1):
    lr = lr * (gamma ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class soft_iou_loss(nn.Module):
    def __init__(self):
        super(soft_iou_loss, self).__init__()

    def forward(self, pred, label):
        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
        unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / (unit + 1e-10))


class FocalLoss2d(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(FocalLoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, prob, target):
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -self.alpha * (pos_weight * torch.log(prob))

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -(1 - self.alpha) * (neg_weight * torch.log(1 - prob))

        loss = (pos_loss + neg_loss)

        return loss.mean()

class weighted_edge_loss(nn.Module):
    def __init__(self, beta_1 = 1, beta_2 = 1):
        super(weighted_edge_loss, self).__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward(self, pred, label):
        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = self.beta_1 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = self.beta_2 * num_positive / (num_positive + num_negative)

        cost = nn.functional.binary_cross_entropy(pred.float(), label.float(), weight=mask, reduction='sum') / (num_negative + num_positive)
        return cost

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice

