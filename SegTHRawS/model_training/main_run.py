
import os
import sys
import torch
import shutil
import logging
import argparse
import lightning as pl
import matplotlib as mpl
from matplotlib import font_manager

import segmentation_models_pytorch as smp
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor

#### RELATIVE IMPORTS
from train_constants import DEVICE, N_CPU
from train_constants import DEFAULT_BS, DEFAULT_LR, DEFAULT_MIN_EPOCHS, DEFAULT_MAX_EPOCHS, DEFAULT_SEED
from train_constants import MAIN_DIRECTORY,DATASETS_PATH, available_models, available_encoders, available_losses

from losses import select_loss
from models import select_model

from training_utils import SegTHRawSModel, SegTHRawSTrainModel
from training_utils import get_training_augmentation, get_preprocessing

from testing_main import main_test
from training_main import main_train

from postprocessing import convert_model_to_onnx
from postprocessing import generate_metrics_plots, generate_model_metrics


sys.path.insert(1,os.path.join(os.path.dirname(os.path.dirname(__file__)),"myriad_conversion"))
from convert_ONNX_to_IR_UNN import convert_ONNX_to_IR_UNN

# Add the Charter font in matplotlib
font_dirs = [os.path.join(os.path.dirname(os.path.dirname(__file__)),'fonts','charter')]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
#Set Charter as default font
mpl.rc('font',family='Charter')



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
    
    parser.add_argument('--gamma',               type=float,
                           help='Importance factor for the focal loss',
                           default=2.0)
    
    parser.add_argument('--alpha',               type=float,
                           help='OPTIONAL: Weight factor associated to false negatives for Focal Tversky Loss',
                           default=0.3)
    
    parser.add_argument('--beta',               type=float,
                           help='OPTIONAL: Weight factor associated to false positives for Focal Tversky Loss',
                           default=0.7)
    
    parser.add_argument('--lr',                 type=float,
                           help='Desired initial learning rate',
                           default=DEFAULT_LR)
    
    parser.add_argument('--lr_scheduler',       type=int,
                           help='Decide if a decreasing lr scheduler is used. Options: 0 or 1. Default: 1',
                           default=1, choices=[0,1])
    
    parser.add_argument('--weakly',                 type=int,
                           help='Decide if weakly segmentation is used. Options: 0 or 1. Default: 1',
                           default=1,choices=[0,1])

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
    ACTIVATION = args.activation
    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights
    loss_name = args.loss
    gamma = args.gamma
    alpha = args.alpha
    beta = args.beta
    lr = args.lr
    lr_scheduler = bool(args.lr_scheduler)
    batch_size = args.bs
    min_epochs = args.min_epochs
    max_epochs = args.max_epochs
    device = args.device
    n_cpu = args.n_cpu
    seed = args.seed
    precision = args.precision
    weakly = bool(args.weakly)

 
    pl.seed_everything(seed,workers=True)

    ACTIVATION_TRAINING = None # To compute the loss during training the activation cannot be used, as the loss requires logits

    if lr_scheduler:
        lr_scheduler_str = '_lr_scheduler'
    else:
        lr_scheduler_str = ''

    if weakly:
        weakly_str = '_weakly'
    else:
        weakly_str = ''


    data_path = os.path.join(datasets_path,f'train_{data_split}_split{weakly_str}_dataset')
    if os.path.isdir(data_path):
        model_name_path = os.path.join(MAIN_DIRECTORY,'models',f'{model_name}_{ENCODER}_fp{precision[:2]}_{loss_name}_gamma{gamma}_alpha{alpha}{lr_scheduler_str}_{data_split}{weakly_str}')
    else:
        raise FileNotFoundError(f"The geographical split folder couldn't be found at: {data_path}")

    os.makedirs(model_name_path,exist_ok=True)
    model_batch_size_path = os.path.join(model_name_path,f'batch_size_{batch_size}')
    model_seed_path = os.path.join(model_batch_size_path,f'seed_{seed}')
    
    os.makedirs(model_batch_size_path,exist_ok=True)
    os.makedirs(model_seed_path,exist_ok=True)


    #Obtains the respective run number, to avoid overwriting previous runs
    run_idx =sum(1 for file in os.listdir(model_seed_path) if file.startswith('run'))

    model_main_path = os.path.join(model_seed_path,f'run_{run_idx}')
    os.makedirs(model_main_path,exist_ok=True)

    metrics_path = os.path.join(model_main_path,'metrics')
    os.makedirs(metrics_path,exist_ok=True)

    myriad_path = os.path.join(model_main_path,'MYRIAD')
    os.makedirs(myriad_path,exist_ok=True)

    logger = CSVLogger(f"{model_main_path}/csv_logs", name=f"{os.path.basename(model_seed_path)}")
    
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    model = select_model(model_name = model_name,
                 ENCODER = ENCODER,
                 ENCODER_WEIGHTS= ENCODER_WEIGHTS,
                 ACTIVATION=ACTIVATION_TRAINING)

    loss = select_loss(loss_name=loss_name,gamma=gamma,alpha=alpha,beta=beta,weakly=weakly)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(model_main_path,'checkpoints'),
            filename=f"{os.path.basename(model_seed_path)}_"+"{epoch}",
            # filename=f"{os.path.basename(model_name_path)}",
            save_top_k=3,
            # every_n_epochs=10,
            monitor="val_loss",
            mode="min",
            verbose=False,
        ),

        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-7,
            patience=8,
            verbose=False,
            mode="min"
        ),

        LearningRateMonitor(
            logging_interval="step"
        )
    ]
    
    if ENCODER and ENCODER_WEIGHTS: # Convert the images into the expected scale
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    else:
        preprocessing_fn = None

    #### NEED TO IMPLEMENT THE AUGMENTATION CLASS IN TRAINING_UTILS
    # augmentation=get_training_augmentation()
    # preprocessing=get_preprocessing(preprocessing_fn)

    model_checkpoint_path = main_train(callbacks = callbacks,
               model =model,
               loss_fn = loss,
               ACTIVATION=ACTIVATION_TRAINING,
               augmentation = get_training_augmentation(),
               preprocessing = get_preprocessing(preprocessing_fn),
               logger = logger,
               images_path = data_path,
               optim_dict = None,
               min_epochs = min_epochs,
               max_epochs = max_epochs,
               lr = lr,
               lr_scheduler=lr_scheduler,
               weakly = weakly,
               batch_size=batch_size,
               device=device,
               num_workers=n_cpu,
               seed = seed,
               precision=precision,
               metrics_path = metrics_path,
               )    

    shutil.copyfile(model_checkpoint_path,os.path.join(model_main_path,os.path.basename(model_checkpoint_path)))
    # shutil.rmtree(os.path.join(model_main_path,'checkpoints'))
    shutil.rmtree(os.path.dirname(model_checkpoint_path))
    
    model_checkpoint_path = os.path.join(model_main_path,os.path.basename(model_checkpoint_path))

    main_test(model_main_path = model_main_path,
              checkpoint_path=model_checkpoint_path,
              model = model,
              loss = loss,
              callbacks = callbacks,
              logger = logger,
              images_path = data_path,
              augmentation= get_training_augmentation(),
              preprocessing = get_preprocessing(preprocessing_fn),
              batch_size=batch_size,
              device=device,
              num_workers=n_cpu,
              seed = seed,
              precision=precision
              )
    
    #Obtain the checkpoint path
    # model_checkpoint_path = [os.path.join(model_main_path,checkpoint_path) for checkpoint_path in os.listdir(model_main_path) if checkpoint_path.endswith('.ckpt')][0]
    
    
    #Create a new model by adding an additional sigmoid layer
    trained_model = SegTHRawSTrainModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path,model=model,loss_fn=loss)
    new_model = SegTHRawSModel(model=trained_model,activation=ACTIVATION)
    
    # os.remove(model_checkpoint_path) #Delete previous checkpoints
    torch.save(new_model,model_checkpoint_path.replace('.ckpt',''))
    model_checkpoint_path = model_checkpoint_path.replace('.ckpt','')
    
    convert_model_to_onnx(model = new_model,
                          model_checkpoint_path=model_checkpoint_path,
                          seed = seed
                          )
    
    # convert_ONNX_to_IR_UNN(models_folder=myriad_path)


    generate_metrics_plots(model_main_path = model_main_path,
                           model_seed_path=model_seed_path,
                           model_checkpoint_path = model_checkpoint_path,
                           metrics_path = metrics_path
                           )
    
    generate_model_metrics(model_name_path=model_name_path)