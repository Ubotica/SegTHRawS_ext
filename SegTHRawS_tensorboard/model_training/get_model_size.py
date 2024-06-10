
"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import argparse
import torch.nn as nn
from models import select_model

def get_model_size(model: nn.Module) -> None:

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024**2
    print(f'Expected model size: {model_size_mb:.3f}MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function that obtains the expected size of the model')

    # add command-line arguments
    
    parser.add_argument('--model_name',         type=str,   help='Name of the desired segmentation model',          default='unet_smp')
    parser.add_argument('--activation',         type=str,   help='Activation function used by the model',           default=None)
    parser.add_argument('--encoder',            type=str,   help='Name of the desired encoder',                     default=None)
    parser.add_argument('--encoder_weights',    type=str,   help='Name of the pre-trained weights of the encoder',  default=None)

    # parse the arguments
    args = parser.parse_args()


    model_name = args.model_name
    ACTIVATION = args.activation
    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights

    model = select_model(model_name = model_name,
                 ENCODER = ENCODER,
                 ENCODER_WEIGHTS= ENCODER_WEIGHTS,
                 ACTIVATION=ACTIVATION)
    
    get_model_size(model)