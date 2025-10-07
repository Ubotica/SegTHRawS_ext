"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""
import segmentation_models_pytorch as smp

from torch import nn

from .train_constants import AVAILABLE_MODELS, AVAILABLE_ENCODERS
from .modified_UNets import (
    UNet,
    small_UNet,
    small_ResUNet,
    small_UNet_3Plus,
    small_Att_UNet,
)


def select_model(
    model_name: str,
    encoder: str = None,
    encoder_weights: str = None,
    in_channels: int = 3,
    classes: int = 1,
    activation: str = None,
    n_filters: int = 32,
    available_models: list = AVAILABLE_MODELS,
    available_encoders: list = AVAILABLE_ENCODERS,
) -> nn.Module:
    """Function for selecting the segmentation model

    Attributes
    ----------
    model_name: str
        Name of the desired model
    encoder: str
        Name of the desired encoder
    encoder_weights: str
        Name of the pre-trained weights for the encoder. Default = None
    in_channels: int
        Number of input channels of the model. Default = 3
    classes: int
        Number of segmentation classes different than empty class. Default = 1
    activation: str
        Final layer activation. Default = None
    n_filters: int
        Number of initial filters of the modified models. Default = 32
    available_models: list
        List of available models. Default = available_models
    available_encoders: list
        List of available encoders. Default = available_encoders

    Outputs
    -------
    model: nn.Module
        Desired PyTorch model

    Notes
    -----

    """

    if model_name not in available_models:
        raise NameError(f"The model {model_name} is not included.")
    if encoder and encoder not in available_encoders:
        raise NameError(f"The encoder {encoder} is not included.")

    if model_name.lower() == "unet_smp":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif model_name.lower() == "deeplabv3plus_smp":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    elif model_name.lower() == "unet++_smp":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    elif model_name.lower() == "unet":
        model = UNet(n_channels=in_channels, n_classes=classes)

    elif model_name.lower() == "mod_unet":
        model = small_UNet(
            n_channels=in_channels, n_classes=classes, n_filters=n_filters
        )

    elif model_name.lower() == "mod_resunet":
        model = small_ResUNet(
            n_channels=in_channels, n_classes=classes, n_filters=n_filters
        )

    elif model_name.lower() == "mod_unet3+":
        model = small_UNet_3Plus(
            in_channels=in_channels, n_classes=classes, n_filters=n_filters
        )

    elif model_name.lower() == "mod_attention_unet":
        model = small_Att_UNet(
            img_ch=in_channels, output_ch=classes, n_filters=n_filters
        )
    else:
        raise FileNotFoundError(f"The model {model_name} couldn't be found")
    return model
