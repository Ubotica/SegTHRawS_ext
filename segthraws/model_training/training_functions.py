"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import lightning as pl
import albumentations as albu
import torchvision.transforms as T

from typing import Any, Union
from torch.nn.modules.loss import _Loss
from lightning.pytorch.tuner import Tuner

from .training_utils import SegTHRawSDataModule, SegTHRawSTrainModel
from .train_constants import N_CPU, DEVICE
from .train_constants import DEFAULT_BS, DEFAULT_LR 
from .train_constants import DEFAULT_MIN_EPOCHS, DEFAULT_MAX_EPOCHS

def train_fn(
        callbacks: list,
        model: pl.LightningModule,
        loss_fn: Any,
        activation: str,
        augmentation: Union[T.Compose, albu.Compose],
        preprocessing: Union[T.Compose, albu.Compose],
        logger: Any,
        images_path: str,
        optim_dict: dict = None,
        min_epochs: int  = DEFAULT_MIN_EPOCHS,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        lr: float = DEFAULT_LR,
        lr_scheduler: bool = True,
        lr_finder: bool = False,
        weakly: bool = True,
        batch_size: int = DEFAULT_BS,
        precision: str = '16-mixed',
        metrics_path: str = None,
        device: str = DEVICE,
        num_workers: int = N_CPU,
        ) -> str:

    """Function for training the input model in Lightining
    
    Attributes
    ----------

    callbacks: list
        Callbacks used for training the model. These include ModelCheckpoint, EarlyStopping, and LearningRateMonitoring
    
    model : nn.Module
            Input segmentation model in PyTorch format
    
    loss_fn : Any
            Loss function
    
    activation: str
            Final layer activation. Default = None
    
    augmentation : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation that applies the image augmentations
    
    preprocessing : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation for data with pre-trained weights 
    
    logger : Any
        Logger function used by Lightning to save the values
    
    images_path : str
        Path to the input images 
    
    optim_dict : dict
            Dictionary for optimizer an learning rate monitoring functions. Default = None
    
    min_epochs: int
        Minimum number of training epochs. Default = DEFAULT_MIN_EPOCHS
    
    max_epochs: int
        Maximum number of training epochs. Default = DEFAULT_MAX_EPOCHS
    
    lr: float
        Learning rate hyperparemeter. Default = DEFAULT_LR
    
    lr_scheduler: bool
        Indicate if the ReduceLROnPlateau learning rate scheduler is used. Default = True
    
    lr_finder: bool
        Indicate if the Lightning's learning rate finder is used. Default = False
    
    weakly: bool 
        Indicate if weakly labelling is used, to modify the metrics. Default = True
    
    batch_size: int
        Batch size hyperparamete r. Default = DEFAULT_BS  
    
    precision: str
        Precision used during training for the output encoding. Default = '16-mixed' 
    
    metrics_path: str
        Path where the metrics plots are saved. Default = None 
    
    device: str
        Indicate if cuda is used. Default = 'cuda'
    
    num_workers: int
        Number of CPU cores. Default = N_CPU

    Outputs
    -------
    model_checkpoint_path: str
        Path to the checkpoint of the best trained model according to the validation loss metric
    
    Notes
    -----

    """

    if device.lower()=='cuda':
        accelerator = 'gpu'
    elif device.lower()=='cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'

    # Trainer
    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator=accelerator,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        num_sanity_val_steps=0,
        precision=precision,
    )

    # Datamodule
    datamodule = SegTHRawSDataModule(
        images_path=images_path,
        augmentation=augmentation,
        preprocessing=preprocessing,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if lr_finder:
        # LightningModule
        lightning_model = SegTHRawSTrainModel(
            model=model,
            loss_fn=loss_fn,
            logger=logger,
            activation=activation,
            optim_dict=optim_dict,
            batch_size=batch_size,
            lr=lr,
            lr_scheduler=lr_scheduler,
            weakly=weakly,
            model_folder_path=os.path.dirname(metrics_path)
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(lightning_model,datamodule=datamodule)

        lr_fig = lr_finder.plot(suggest=True)
        lr_fig.savefig(os.path.join(metrics_path,'lr_finder'))
        lightning_model.hparams.lr = lr_finder.suggestion()
    
    else:
        # LightningModule
        lightning_model = SegTHRawSTrainModel(
            model=model,
            loss_fn=loss_fn,
            logger=logger,
            activation=activation,
            optim_dict=optim_dict,
            batch_size=batch_size,
            lr=lr,
            lr_scheduler=lr_scheduler,
            weakly=weakly,
            model_folder_path=os.path.dirname(metrics_path)
            )


    # Start training
    trainer.fit(model=lightning_model, datamodule=datamodule)

    return callbacks[0].best_model_path

def retrain_fn(
        lightning_model: pl.LightningModule,
        callbacks: list,
        model: pl.LightningModule,
        loss_fn: Any,
        activation: str,
        augmentation: Union[T.Compose, albu.Compose],
        preprocessing: Union[T.Compose, albu.Compose],
        logger: Any,
        images_path: str,
        optim_dict: dict = None,
        min_epochs: int  = DEFAULT_MIN_EPOCHS,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        lr: float = DEFAULT_LR,
        lr_scheduler: bool = True,
        lr_finder: bool = False,
        weakly: bool = True,
        batch_size: int = DEFAULT_BS,
        precision: str = '16-mixed',
        metrics_path: str = None,
        device: str = DEVICE,
        num_workers: int = N_CPU,
        ) -> str:

    """Function for retraining models
    
    Attributes
    ----------

    lightning_model : pl.LightningModule,
        Model to be retrained.

    datasets_path : str,
        Path to the directory of the training datasets.  Default=DATASETS_PATH
    
    data_split : str,
        help="Define the training split for the dataset. Default='geo', choices=['geo','random']
    
    band_list : str,
        Specify the band combination for the training dataset.  Default=["B12","B11","B8A"]

    model_name : str,
        Name of the desired segmentation model.  Default='unet_smp', choices=available_models
    
    activation : str,
        Final activation layer of the model. Default='sigmoid'
    
    encoder : str,
        Name of the desired encoder. Default='mobilenet_v2', choices=available_encoders
    
    encoder_weights : str,
        Name of the pre-trained weights of the encoder. Default='imagenet'
    
    n_filters : int,
        Number of filters for the U-Net modification. Default=32
    
    loss : str,
        Name of the desired loss function. Default='focal_loss_smp', choices=available_losses
    
    gamma : float,
        Importance factor for the focal loss. Default=2.0
    
    alpha : float,
        OPTIONAL: Weight factor associated to class weight in Focal Loss or to false negatives in Focal Tversky Loss. Default=0.25
    
    beta : float,
        OPTIONAL: Weight factor associated to false positives for Focal Tversky Loss. Default=0.7
    
    lr : float,
        Desired initial learning rate.  Default=DEFAULT_LR 

    lr_scheduler : bool,
        Decide if a decreasing lr scheduler is used.  Default=True
    lr_finder : bool,
        Select if a learning finder is used. Default=True
    
    weakly : bool,
        Decide if weakly segmentation is used. Default=True

    batch_size : int,
        Desired batch size.  Default=DEFAULT_BS
    
    min_epochs : int,
        Number of minimum epochs for training the model. Default=DEFAULT_MIN_EPOCHS
    
    max_epochs : int,
        Number of maximum epochs for training the model. Default=DEFAULT_MAX_EPOCHS
    
    device : str,
        Device used for training the model. Options: 'cuda' or 'cpu. Default=DEVICE, choices = ['cuda', 'cpu']
    
    n_cpu : int,
        Define the number of CPU cores to be used. Default is max number per device : N_CPU
    
    seed : int,
        Seed used for training the model. Default: 42. Default=DEFAULT_SEED
    
    precision : str,
        Specify whether mixed precision is used. Default: 16-mixed. Default='16-mixed',choices=['16-mixed','32']

    Outputs
    -------
    Trained model in PyTorch and ONNX formats, and the training and testing logs 

    Notes
    -----

    """


    if device.lower()=='cuda':
        accelerator = 'gpu'
    elif device.lower()=='cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'

    # Trainer
    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator=accelerator,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        num_sanity_val_steps=0,
        precision=precision,
        )

    # Datamodule
    datamodule = SegTHRawSDataModule(
        images_path=images_path,
        augmentation=augmentation,
        preprocessing=preprocessing,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if lr_finder:

        # LightningModule
        old_lightning_model = SegTHRawSTrainModel(
            model=model,
            loss_fn=loss_fn,
            logger=logger,
            activation=activation,
            optim_dict=optim_dict,
            batch_size=batch_size,
            lr=lr,
            lr_scheduler=lr_scheduler,
            weakly=weakly,
            model_folder_path=os.path.dirname(metrics_path)
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(old_lightning_model,datamodule=datamodule)

        lr_fig = lr_finder.plot(suggest=True)
        lr_fig.savefig(os.path.join(metrics_path,'lr_finder'))
        lightning_model.hparams.lr = lr_finder.suggestion()
    else:
        lightning_model.hparams.lr = lr

    # Start training
    trainer.fit(model=lightning_model, datamodule=datamodule)

    return callbacks[0].best_model_path

def test_fn(
        checkpoint_path: str,
        model: pl.LightningModule,
        loss: _Loss,
        callbacks: list,
        logger: Any,
        images_path: str,
        augmentation: Union[T.Compose, albu.Compose],
        preprocessing: Any,
        batch_size: int = DEFAULT_BS,
        precision: str = '16-mixed',
        device: str = DEVICE,
        num_workers: int = N_CPU,
        save_imgs : bool = True,
        ) -> None:

    """Function for performing the testing of the input model in Lightining

    Attributes
    ----------

    checkpoint_path : str
        Path to the checkpoint of the model to be tested
    
    model : nn.Module
        Input segmentation model in PyTorch format
    
    loss : _Loss
        Loss function
    
    callbacks: list
        Callbacks used for training the model. These include ModelCheckpoint, EarlyStopping, and LearningRateMonitoring
    
    logger : Any
        Logger function used by Lightning to save the values
    
    images_path : str
        Path to the input images 
    
    augmentation : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation that applies the image augmentations
    
    preprocessing : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation for data with pre-trained weights 
    
    batch_size: int
        Batch size hyperparamete r. Default = DEFAULT_BS  
    
    precision: str
        Precision used during training for the output encoding. Default = '16-mixed' 
    
    device: str
        Indicate if cuda is used. Default = 'cuda'
    
    num_workers: int
        Number of CPU cores. Default = N_CPU
    
    save_imgs: bool
        Determine whether the testing output masks are saved. Default = True
    Outputs
    -------

    Notes
    -----

    """

    if device.lower()=='cuda':
        accelerator = 'gpu'
    elif device.lower()=='cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'


    trained_model = SegTHRawSTrainModel.load_from_checkpoint(checkpoint_path=checkpoint_path,model=model,loss_fn=loss, save_imgs =save_imgs)
    trained_model.eval();

    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator=accelerator,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=1,
        min_epochs=1,
        num_sanity_val_steps=0,
        precision=precision,
    )

    # Datamodule
    datamodule = SegTHRawSDataModule(
        images_path=images_path,
        augmentation=augmentation,
        preprocessing=preprocessing,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer.test(model=trained_model,datamodule=datamodule)[0];

def predict_fn(
        checkpoint_path: str,
        model: pl.LightningModule,
        loss: _Loss,
        callbacks: list,
        logger: Any,
        images_path: str,
        augmentation: Union[T.Compose, albu.Compose],
        preprocessing: Union[T.Compose, albu.Compose],
        batch_size: int = DEFAULT_BS,
        precision: str = '16-mixed',
        device: str = DEVICE,
        num_workers: int = N_CPU,
        ) -> None:
    """Function for performing the testing of the input model in Lightining

    Attributes
    ----------

    checkpoint_path : str
        Path to the checkpoint of the model to be tested
    
    model : nn.Module
        Input segmentation model in PyTorch format
    
    loss : _Loss
        Loss function
    
    callbacks: list
        Callbacks used for training the model. These include ModelCheckpoint, EarlyStopping, and LearningRateMonitoring
    
    logger : Any
        Logger function used by Lightning to save the values
    
    images_path : str
        Path to the input images 
    
    augmentation : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation that applies the image augmentations
    
    preprocessing : Union[T.Compose, albu.Compose]
        albumentations.Compose transformation for data with pre-trained weights 
    
    batch_size: int
        Batch size hyperparamete r. Default = DEFAULT_BS  
    
    precision: str
        Precision used during training for the output encoding. Default = '16-mixed' 
    
    device: str
        Indicate if cuda is used. Default = 'cuda'
    
    num_workers: int
        Number of CPU cores. Default = N_CPU

    Outputs
    -------

    Notes
    -----

    """


    if device.lower()=='cuda':
        accelerator = 'gpu'
    elif device.lower()=='cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'


    trained_model = SegTHRawSTrainModel.load_from_checkpoint(checkpoint_path=checkpoint_path,model=model,loss_fn=loss)
    trained_model.eval();

    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator=accelerator,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=1,
        min_epochs=1,
        num_sanity_val_steps=0,
        precision=precision,
    )

    # Datamodule
    datamodule = SegTHRawSDataModule(
        images_path=images_path,
        augmentation=augmentation,
        preprocessing=preprocessing,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer.predict(model=trained_model,datamodule=datamodule);