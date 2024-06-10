"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import torch

MAIN_DIRECTORY: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# DATA_PATH: str = os.path.join(MAIN_DIRECTORY,'datasets','train_geo_split_dataset')

DATASETS_PATH: str = os.path.join(MAIN_DIRECTORY,'datasets')


DEFAULT_LR = 0.0003
DEFAULT_BS = 16
DEFAULT_MIN_EPOCHS = 150
DEFAULT_MAX_EPOCHS = 200
DEFAULT_SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CPU = os.cpu_count()

OPSET_VERSION = 15

OPENVINO_IMAGE = 'openvino/ubuntu20_dev:2022.1.0'
CONTAINER_NAME = 'openvino_2022.1'


available_models =['unet_smp','deeplabv3plus_smp','unet++_smp']
available_encoders =['resnet18','resnet34','resnext50_32x4d','timm-resnest14d','timm-resnest26d',
                     'se_resnet50',
                     'densenet121','densenet169','densenet201',
                     'xception',
                     'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5',
                     'mobilenet_v2','timm-mobilenetv3_large_100',
                     'dpn68',
                     'vgg16','vgg16_bn',
                     'mobileone_s0','mobileone_s1','mobileone_s2','mobileone_s3','mobileone_s4']
available_losses =['focal_loss_smp','dice_loss_smp','jaccard_loss_smp','combined_focal_dice_loss_smp','tversky_loss','focal_tversky_loss','tversky_loss_smp']
available_metrics =[]


mean_imagenet_imgs =  [0.485, 0.456, 0.406]
std_imagenet_imgs =  [0.229, 0.224, 0.225]


optim_dict = None

