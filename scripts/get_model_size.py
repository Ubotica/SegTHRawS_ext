"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import sys
import argparse
import torch.nn as nn

sys.path.insert(1,os.path.dirname(os.path.dirname(__file__)))

from segthraws.model_training.models import select_model

def get_model_size(model: nn.Module,
                   output_format:str ='MB'
                   ) -> None:
    """Function for obtaining the size of a model

    Attributes
    ----------

    model: Union[nn.Module, pl.LightningModule]
        Desired model to measure its number of parameters or size 

    output_format: str
        String that idicates the desired whether the model size is expressed in MB or in number of parameteres 

    Outputs
    -------
    Print the expected model size in MB or in number of parameters

    Notes
    -----

    """
    if output_format.lower() == 'mb':
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / 1024**2
        print(f'Expected model size: {model_size_mb:.3f}MB')
    else:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() 
        model_size = param_size + buffer_size 
        print(f'Total number of parameters: {model_size}')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function that obtains the expected size of the model')

    
    parser.add_argument('--model_name',         type=str,   help='Name of the desired segmentation model',          default='unet_smp')
    parser.add_argument('--activation',         type=str,   help='Activation function used by the model',           default=None)
    parser.add_argument('--encoder',            type=str,   help='Name of the desired encoder',                     default="mobilenet_v2")
    parser.add_argument('--encoder_weights',    type=str,   help='Name of the pre-trained weights of the encoder',  default=None)
    parser.add_argument('--n_filters',          type=int,   help='Number of filter used for the modified U-Net network',  default=32)
    parser.add_argument('--output_format',      type=str,   help='Determine the output format: MB or number of parameters. Options: "MB" OR "params"',  default="params",choices=["MB","params"])

    # parse the arguments
    args = parser.parse_args()


    model_name = args.model_name
    activation = args.activation
    encoder = args.encoder
    encoder_weights = args.encoder_weights
    n_filters = args.n_filters
    output_format = args.output_format

    if encoder_weights and encoder_weights.lower()=="none":
        encoder_weights = None

    model = select_model(model_name = model_name,
                 encoder = encoder,
                 encoder_weights= encoder_weights,
                 n_filters = n_filters,
                 activation=activation)
    
    get_model_size(model,output_format=output_format)