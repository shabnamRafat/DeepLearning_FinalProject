
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        N, C = input.size(0), input.size(1)
        input_soft = torch.softmax(input, dim=1)
        target_one_hot = torch.zeros_like(input_soft)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        input_flat = input_soft.view(N, C, -1)
        target_flat = target_one_hot.view(N, C, -1)
        intersection = (input_flat * target_flat).sum(dim=2)
        union = input_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, input, target):
        return self.dice_weight * self.dice_loss(input, target) + \
               self.focal_weight * self.focal_loss(input, target)