import os
import sys
# sys.path.insert(1,'..')
sys.path.insert(1,os.path.dirname(os.path.dirname(__file__)))

from segthraws.model_training.main_run import main_run

from segthraws.model_training.train_constants import DATASETS_PATH, available_models, available_encoders, available_losses
from segthraws.model_training.train_constants import DEFAULT_BS, DEFAULT_LR, DEFAULT_MIN_EPOCHS, DEFAULT_MAX_EPOCHS, DEFAULT_SEED
from segthraws.model_training.train_constants import DEVICE, N_CPU

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SegTHRawS model training')

    # add command-line arguments
    
    
    parser.add_argument('--datasets_path',      type=str,
                           help='Path to the directory of the training datasets',
                           default=DATASETS_PATH)
    
    parser.add_argument('--data_split',         type=str,
                           help="Define the training split for the dataset. Options: 'geo' or 'random'",
                           default='geo', choices=['geo','random'])
    
    parser.add_argument('--model_name',         type=str,
                           help='Name of the desired segmentation model',
                           default='unet_smp', choices=available_models)
    
    parser.add_argument('--activation',         type=str,
                           help='Final activation layer of the model',
                           default='sigmoid')
    
    parser.add_argument('--encoder',            type=str,
                           help='Name of the desired encoder',
                           default='mobilenet_v2', choices=available_encoders)
    
    parser.add_argument('--encoder_weights',    type=str,
                           help='Name of the pre-trained weights of the encoder',
                           default='imagenet')
    
    parser.add_argument('--loss',               type=str,
                           help='Name of the desired loss function',
                           default='focal_loss_smp', choices=available_losses)
    
    parser.add_argument('--lr',                 type=float,
                           help='Desired initial learning rate',
                           default=DEFAULT_LR)
    
    parser.add_argument('--bs',                 type=int,
                           help='Desired batch size',
                           default=DEFAULT_BS)
    
    parser.add_argument('--min_epochs',         type=int,
                           help='Number of minimum epochs for training the model',
                           default=DEFAULT_MIN_EPOCHS)
    
    parser.add_argument('--max_epochs',         type=int,
                           help='Number of maximum epochs for training the model',
                           default=DEFAULT_MAX_EPOCHS)
    
    parser.add_argument('--device',             type=str,
                           help="Device used for training the model. Options: 'cuda' or 'cpu'",
                           default=DEVICE)
    
    parser.add_argument('--n_cpu',              type=int,
                           help='Define the number of CPU cores to be used. Default is max number per device',
                           default=N_CPU)
    
    parser.add_argument('--seed',               type=int,
                           help='Seed used for training the model. Default: 42',
                           default=DEFAULT_SEED)
    
    parser.add_argument('--precision',          type=str,
                           help='Specify whether mixed precision is used. Default: 16-mixed',
                           default='16-mixed',choices=['16-mixed','32'])

    # parse the arguments
    args = parser.parse_args()

    datasets_path = args.datasets_path
    data_split = args.data_split
    model_name = args.model_name
    activation = args.activation
    encoder = args.encoder
    encoder_weights = args.encoder_weights
    loss_name = args.loss
    lr = args.lr
    batch_size = args.bs
    min_epochs = args.min_epochs
    max_epochs = args.max_epochs
    device = args.device
    n_cpu = args.n_cpu
    seed = args.seed
    precision = args.precision

    main_run(datasets_path = datasets_path,
             data_split = data_split,
             model_name = model_name,
             activation = activation,
             encoder = encoder,
             encoder_weights = encoder_weights,
             loss_name = loss_name,
             lr = lr,
             batch_size = batch_size,
             min_epochs = min_epochs,
             max_epochs = max_epochs,
             device = device,
             n_cpu = n_cpu,
             seed = seed,
             precision = precision,
             )