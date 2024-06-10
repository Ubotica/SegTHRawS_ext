"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""
import torch.nn as nn
import lightning as pl
import segmentation_models_pytorch as smp

from typing import Any, Union
from train_constants import available_models, available_encoders

def select_model(model_name: str,
                 ENCODER: str,
                 ENCODER_WEIGHTS: str,
                 IN_CHANNELS: int = 3,
                 CLASSES: int = 1,
                 ACTIVATION: str = None,
                 available_models: list = available_models,
                 available_encoders: list = available_encoders) -> nn.Module:
    
    #### Create a models list and encoders list in constants, to ensure that the input model and encoder are available in this.

    if model_name not in available_models:
        raise NameError(f'The model {model_name} is not included.')
    if ENCODER and ENCODER not in available_encoders:
        raise NameError(f'The encoder {ENCODER} is not included.')
    

    if model_name.lower() == 'unet_smp':
        model = smp.Unet(encoder_name=ENCODER, 
                         encoder_weights=ENCODER_WEIGHTS, 
                         in_channels = IN_CHANNELS,
                         classes=CLASSES, 
                         activation=ACTIVATION
        )
    if model_name.lower() == 'deeplabv3plus_smp':
        model = smp.DeepLabV3Plus(encoder_name=ENCODER, 
                         encoder_weights=ENCODER_WEIGHTS, 
                         in_channels = IN_CHANNELS,
                         classes=CLASSES, 
                         activation=ACTIVATION
        )

    if model_name.lower() == 'unet++_smp':
        model = smp.UnetPlusPlus(encoder_name=ENCODER, 
                         encoder_weights=ENCODER_WEIGHTS, 
                         in_channels = IN_CHANNELS,
                         classes=CLASSES, 
                         activation=ACTIVATION
        )


    return model


def get_model_size(model: Union[nn.Module, pl.LightningModule]) -> None:

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024**2
    print(f'Expected model size: {model_size_mb:.3f}MB')
