import torch
import warnings
import numpy as np
import lightning as pl

from typing import Any
from torch.nn.modules.loss import _Loss 
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss, JaccardLoss, TverskyLoss

from train_constants import available_losses

class Combined_Focal_Dice_Loss(pl.LightningModule):
    '''
    Combined weighted loss between Focal Loss and Dice Loss  
    '''
    def __init__(self,
                 focal_loss_weight: float = 0.5,
                 dice_weight: float = None,
                 log_dice_loss: bool = False):
        
        super(Combined_Focal_Dice_Loss, self).__init__()
        
        self.focal_loss_weight = focal_loss_weight
        self.dice_weight = (1 - focal_loss_weight) if dice_weight is None else dice_weight

        if self.focal_loss_weight + self.dice_weight != 1:
            warnings.warn("Sum of Focal and Dice loss weights is not 1.0: "
                          f"{self.focal_loss_weight:.2f} + {self.dice_weight:.2f} = "
                          f"{self.focal_loss_weight + self.dice_weight:.2f}")

        self.log_dice_loss = log_dice_loss


    # def dice_score(y_pred, y_true, eps=1e-15, smooth=1.):
    #     intersection = (y_pred * y_true).sum()
    #     union = y_pred.sum() + y_true.sum()
    #     return (2. * intersection + smooth) / (union + smooth + eps)


    def forward(self, y_pred, y_true):

        focal_loss_fn = FocalLoss(mode= 'binary')
        focal_loss_fn.__name__ = 'focal_loss'
        dice_loss_fn = DiceLoss(mode= 'binary',from_logits=True,log_loss=self.log_dice_loss) #Typically Dice use the masks and not logits, that is why from logits is used because y_pred are the logits
        dice_loss_fn.__name__ = 'dice_loss'


        focal_loss = focal_loss_fn(y_pred, y_true) 
        dice_loss = dice_loss_fn(y_pred, y_true) 
        
        # y_pred = torch.sigmoid(y_pred)
        # dice_loss = 1- dice_score(y_pred, y_true)
        # log_dice_loss = -torch.log(dice_score(y_pred, y_true))
        
        loss = self.focal_loss_weight * focal_loss + self.dice_weight * dice_loss

        return loss


class Tversky_Loss(pl.LightningModule):
    '''
    Tversky and Focal Tversky Loss functions  
    '''
    def __init__(self,
                 alpha: float = 0.3,
                 eps: float = 1e-7,
                 beta: float = 0.7,
                 gamma: float = None,
                 ):
        
        super(Tversky_Loss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def forward(self, y_pred, y_true):

        # print('BEFORE: ',y_pred[0])

        # y_pred = torch.sigmoid(y_pred)>0.5
        y_pred = torch.round(torch.sigmoid(y_pred))

        # print('AFTER: ',y_pred[0])


        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1.0-y_true))
        fn = torch.sum((1-y_pred) * y_true)

        # tp = torch.sum(torch.logical_and(y_pred==1,y_true==1))
        # fp = torch.sum(torch.logical_and(y_pred==1,y_true==0))
        # fn = torch.sum(torch.logical_and(y_pred==0,y_true==1))

        tversky_idx= (tp/(tp+ self.alpha*fn + self.beta*fp)).clamp_min(self.eps)

        loss = 1-tversky_idx

        if self.gamma: #Focal Tversky Loss selected
            loss = loss**(1/self.gamma)
            # loss = loss**(self.gamma)

        return loss


def select_loss(loss_name: str,
                available_losses: list = available_losses,
                gamma:float = None,
                alpha: float = 2.0,
                beta: float = None,
                weakly: bool = False,
                ) -> _Loss:
    
    if loss_name not in available_losses:
        raise NameError(f'The loss function {loss_name} is not included.')

    else:
        if weakly: 
            ignore_idx=-1 
        else: 
            ignore_idx=None
        if loss_name.lower()=='focal_loss_smp':
            if gamma: 
                loss = FocalLoss(mode= 'binary',gamma=gamma,alpha=alpha,ignore_index=ignore_idx)
            else:
                loss = FocalLoss(mode= 'binary',alpha=alpha,ignore_index=ignore_idx)
            loss.__name__ = 'focal_loss'

        if loss_name.lower()=='dice_loss_smp':
            loss = DiceLoss(mode= 'binary',ignore_index=ignore_idx)
            loss.__name__ = 'dice_loss'
        
        if loss_name.lower()=='jaccard_loss_smp':
            loss = JaccardLoss(mode= 'binary')
            loss.__name__ = 'jaccard_loss'

        if loss_name.lower()=='combined_focal_dice_loss_smp':
            loss = Combined_Focal_Dice_Loss()

        if loss_name.lower()=='tversky_loss_smp':
            loss = TverskyLoss(mode='binary',alpha=alpha,beta=beta,gamma=gamma)
            loss.__name__ = 'tversky_loss'

        
        if loss_name.lower()=='tversky_loss':
            loss = Tversky_Loss(alpha=alpha,beta=beta)
        if loss_name.lower()=='focal_tversky_loss':
            loss = Tversky_Loss(alpha=alpha,beta=beta,gamma=gamma)

    return loss
