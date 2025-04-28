import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1.):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))

        bcedice_loss = dice_loss + bce_loss
        return bcedice_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth))

        return dice_loss
