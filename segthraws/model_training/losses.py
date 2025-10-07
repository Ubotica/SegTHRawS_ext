"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import warnings

import torch
import lightning as pl

from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses import (
    FocalLoss,
    DiceLoss,
    JaccardLoss,
    TverskyLoss,
)

from .train_constants import AVAILABLE_LOSSES


class combined_focal_dice_loss(pl.LightningModule):
    """
    Combined weighted loss between Focal Loss and Dice Loss
    """

    def __init__(
        self,
        focal_loss_weight: float = 0.5,
        dice_weight: float = None,
        log_dice_loss: bool = False,
    ):

        super(combined_focal_dice_loss, self).__init__()

        self.focal_loss_weight = focal_loss_weight
        self.dice_weight = (
            (1 - focal_loss_weight) if dice_weight is None else dice_weight
        )

        if self.focal_loss_weight + self.dice_weight != 1:
            warnings.warn(
                "Sum of Focal and Dice loss weights is not 1.0: "
                f"{self.focal_loss_weight:.2f} + {self.dice_weight:.2f} = "
                f"{self.focal_loss_weight + self.dice_weight:.2f}"
            )

        self.log_dice_loss = log_dice_loss

    def forward(self, y_pred, y_true):

        focal_loss_fn = FocalLoss(mode="binary")
        focal_loss_fn.__name__ = "focal_loss"
        dice_loss_fn = DiceLoss(
            mode="binary", from_logits=True, log_loss=self.log_dice_loss
        )  # Typically Dice use the masks and not logits. y_pred are the logits
        dice_loss_fn.__name__ = "dice_loss"

        focal_loss = focal_loss_fn(y_pred, y_true)
        dice_loss = dice_loss_fn(y_pred, y_true)

        loss = self.focal_loss_weight * focal_loss + self.dice_weight * dice_loss

        return loss


class tversky_loss(pl.LightningModule):
    """
    Tversky and Focal Tversky Loss functions
    """

    def __init__(
        self,
        alpha: float = 0.3,
        eps: float = 1e-7,
        beta: float = 0.7,
        gamma: float = None,
    ):

        super(tversky_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma

    def forward(self, y_pred, y_true):

        y_pred = torch.round(torch.sigmoid(y_pred))

        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1.0 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)

        tversky_idx = (tp / (tp + self.alpha * fn +
                       self.beta * fp)).clamp_min(self.eps)

        loss = 1 - tversky_idx

        if self.gamma:  # Focal Tversky Loss selected
            loss = loss ** (1 / self.gamma)
            # loss = loss**(self.gamma)

        return loss


def select_loss(
    loss_name: str,
    available_losses: list = AVAILABLE_LOSSES,
    gamma: float = None,
    alpha: float = 2.0,
    beta: float = None,
    weakly: bool = False,
) -> _Loss:
    """Function for selecting the loss function

    Attributes
    ----------
    loss_name: str
        Name of the desired loss function
    available_losses: list
        List of available loss functions. This ensures that the input loss function is 
        supported by the pipeline. Default = available_losses
    gamma: float
        Focal Loss parameter that determines the importance of the "hard" to segment cases.
    alpha: float
        Focal Loss parameter that determines the importance factor of the positve class. 
        It can also be the first Tversky Loss parameter that is associated with the 
        importance factor of the False Positives
    beta: float
        Second Tversky Loss parameter, associated with the importance factor of the False Negatives
    weakly: bool
        Indicate if weakly labelling is used, to modify the metrics. Default = True

    Outputs
    -------
    loss_function: _Loss
        Desired Loss function

    Notes
    -----

    """
    if loss_name not in available_losses:
        raise NameError(f"The loss function {loss_name} is not included.")

    if weakly:
        ignore_idx = -1
    else:
        ignore_idx = None
    
    if loss_name.lower() == "focal_loss_smp":
        if gamma:
            loss = FocalLoss(
                mode="binary", gamma=gamma, alpha=alpha, ignore_index=ignore_idx
            )
        else:
            loss = FocalLoss(mode="binary", alpha=alpha,
                                ignore_index=ignore_idx)
        loss.__name__ = "focal_loss"

    elif loss_name.lower() == "dice_loss_smp":
        loss = DiceLoss(mode="binary", ignore_index=ignore_idx)
        loss.__name__ = "dice_loss"

    elif loss_name.lower() == "jaccard_loss_smp":
        loss = JaccardLoss(mode="binary")
        loss.__name__ = "jaccard_loss"

    elif loss_name.lower() == "combined_focal_dice_loss_smp":
        loss = combined_focal_dice_loss()

    elif loss_name.lower() == "tversky_loss_smp":
        loss = TverskyLoss(
            mode="binary",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ignore_index=ignore_idx,
        )
        loss.__name__ = "tversky_loss"

    elif loss_name.lower() == "tversky_loss":
        loss = tversky_loss(alpha=alpha, beta=beta)
    elif loss_name.lower() == "focal_tversky_loss":
        loss = tversky_loss(alpha=alpha, beta=beta, gamma=gamma)

    else:
        raise FileNotFoundError(f"Loss function {loss_name} couldn't be found.")

    return loss
