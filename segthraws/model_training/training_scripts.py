"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details

 - This sript includes the main execution file for training, testing and 
    retraining segmentation models.

 - It was designed to adapt multiple input combinations

 - The main outputs are the trained models in both PyTorch and ONNX format,
    as well as the training and testing TensorBoard logs
"""

import os
import re
import shutil
import logging

import torch
import lightning as pl
import matplotlib as mpl
from matplotlib import font_manager

import segmentation_models_pytorch as smp
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# RELATIVE IMPORTS
from .train_constants import DEVICE, N_CPU
from .train_constants import (
    DEFAULT_BS,
    DEFAULT_LR,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_SEED,
)
from .train_constants import MAIN_DIRECTORY, DATASETS_PATH

from .losses import select_loss
from .models import select_model

from .training_utils import SegTHRawSModel, SegTHRawSTrainModel
from .training_utils import get_training_augmentation, get_preprocessing

from .training_functions import train_fn, test_fn, retrain_fn

from .postprocessing import convert_model_to_onnx


# Add the Charter font in matplotlib
font_dirs = [
    os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "fonts", "charter")
]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
# Set Charter as default font
mpl.rc("font", family="Charter")


def model_train_test(
    datasets_path: str = DATASETS_PATH,
    data_split: str = "geo",
    band_list_dataset: str = ["B12", "B11", "B8A"],
    model_name: str = "unet_smp",
    activation: str = "sigmoid",
    encoder: str = "mobilenet_v2",
    encoder_weights: str = "imagenet",
    n_filters: int = 32,
    loss_name: str = "focal_loss_smp",
    gamma: float = 2.0,
    alpha: float = 0.25,
    beta: float = 0.7,
    lr: float = DEFAULT_LR,
    lr_scheduler: int = 1,
    lr_finder: int = 1,
    weakly: int = 1,
    batch_size: int = DEFAULT_BS,
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    device: str = DEVICE,
    n_cpu: int = N_CPU,
    seed: int = DEFAULT_SEED,
    precision: str = "16-mixed",
) -> None:
    """

    Attributes
    ----------

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
        OPTIONAL: Weight factor associated to class weight in Focal Loss 
        or to false negatives in Focal Tversky Loss. Default=0.25

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
        Device used for training the model. Options: 'cuda' or 'cpu.
        Default=DEVICE, choices = ['cuda', 'cpu']

    n_cpu : int,
        Define the number of CPU cores to be used. Default is max number per device : N_CPU

    seed : int,
        Seed used for training the model. Default: 42. Default=DEFAULT_SEED

    precision : str,
        Specify whether mixed precision is used. Default: 16-mixed.
        Default='16-mixed',choices=['16-mixed','32']

    Outputs
    -------
    Trained model in PyTorch and ONNX formats, and the training and testing logs

    Notes
    -----

    """

    pl.seed_everything(seed, workers=True)

    # To compute the loss during training the activation cannot be used, as the loss requires logits
    activation_training = None

    band_list_str = "_".join(band_list_dataset)

    if lr_scheduler:
        lr_scheduler_str = "_lr_scheduler"
    else:
        lr_scheduler_str = ""

    if weakly:
        weakly_str = "_weakly"
    else:
        weakly_str = ""

    if encoder_weights.lower() == "none":
        encoder_weights = None
        encoder_weights_str = ""
    else:
        encoder_weights_str = f"_{encoder_weights}"

    if encoder.lower() == "none":
        encoder = None
        encoder_str = ""
    else:
        encoder_str = f"_{encoder}"

    n_filters_str = f"_{n_filters}"

    if lr_finder:
        lr_str = "_lr_finder"
    else:
        lr_str = f"_lr{lr}"

    data_path = os.path.join(
        datasets_path, f"train_{data_split}_split{weakly_str}_{band_list_str}_dataset"
    )
    if os.path.isdir(data_path):
        filename_parts = [
            f"{model_name}{encoder_str}{encoder_weights_str}{n_filters_str}",
            f"fp{precision[:2]}",
            loss_name,
            f"gamma{gamma}",
            f"alpha{alpha}",
            f"{lr_str}{lr_scheduler_str}",
            f"{data_split}{weakly_str}",
            band_list_str,
        ]

        # Join only the non-empty parts with an underscore
        filename = "_".join(part for part in filename_parts if part)

        # Use the modern pathlib to create the full path
        model_name_path = os.path.join(MAIN_DIRECTORY, "models",filename)

    else:
        raise FileNotFoundError(
            f"The training dataset folder couldn't be found at: {data_path}"
        )

    os.makedirs(model_name_path, exist_ok=True)
    model_batch_size_path = os.path.join(
        model_name_path, f"batch_size_{batch_size}")
    model_seed_path = os.path.join(model_batch_size_path, f"seed_{seed}")

    os.makedirs(model_batch_size_path, exist_ok=True)
    os.makedirs(model_seed_path, exist_ok=True)

    # Obtains the respective run number, to avoid overwriting previous runs
    run_idx = sum(1 for file in os.listdir(
        model_seed_path) if file.startswith("run"))

    model_main_path = os.path.join(model_seed_path, f"run_{run_idx}")
    os.makedirs(model_main_path, exist_ok=True)

    metrics_path = os.path.join(model_main_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    myriad_path = os.path.join(model_main_path, "MYRIAD")
    os.makedirs(myriad_path, exist_ok=True)

    # logger = CSVLogger(f"{model_main_path}/csv_logs", name=f"{os.path.basename(model_seed_path)}")

    logger = TensorBoardLogger(
        f"{model_main_path}/logs",
        name=f"{os.path.basename(model_seed_path)}",
        default_hp_metric=False,
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
        logging.WARNING)

    model = select_model(
        model_name=model_name,
        encoder=encoder,
        encoder_weights=encoder_weights,
        activation=activation_training,
        n_filters=n_filters,
    )

    loss = select_loss(
        loss_name=loss_name, gamma=gamma, alpha=alpha, beta=beta, weakly=weakly
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(model_main_path, "checkpoints"),
            filename=f"{os.path.basename(model_seed_path)}_" + "{epoch}",
            # filename=f"{os.path.basename(model_name_path)}",
            save_top_k=3,
            # every_n_epochs=10,
            monitor="val_loss",
            mode="min",
            verbose=False,
        ),
        EarlyStopping(
            monitor="val_loss", min_delta=1e-7, patience=8, verbose=False, mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if encoder and encoder_weights:  # Convert the images into the expected scale
        print(encoder, encoder_weights)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)
    else:
        preprocessing_fn = None

    model_checkpoint_path = train_fn(
        callbacks=callbacks,
        model=model,
        loss_fn=loss,
        activation=activation_training,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        logger=logger,
        images_path=data_path,
        optim_dict=None,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_finder=lr_finder,
        weakly=weakly,
        batch_size=batch_size,
        device=device,
        num_workers=n_cpu,
        precision=precision,
        metrics_path=metrics_path,
    )

    # Move the final model checkpoint to the main model directory to the
    shutil.copyfile(
        model_checkpoint_path,
        os.path.join(model_main_path, os.path.basename(model_checkpoint_path)),
    )
    shutil.rmtree(os.path.dirname(model_checkpoint_path))
    model_checkpoint_path = os.path.join(
        model_main_path, os.path.basename(model_checkpoint_path)
    )

    # Testing
    test_fn(
        model_main_path=model_main_path,
        checkpoint_path=model_checkpoint_path,
        model=model,
        loss=loss,
        callbacks=callbacks,
        logger=logger,
        images_path=data_path,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        batch_size=batch_size,
        device=device,
        num_workers=n_cpu,
        precision=precision,
    )

    # Create the inference model by adding an additional sigmoid layer
    trained_model = SegTHRawSTrainModel.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path, model=model, loss_fn=loss
    )
    new_model = SegTHRawSModel(model=trained_model, activation=activation)

    # Save new model
    torch.save(new_model, model_checkpoint_path.replace(".ckpt", ""))
    model_checkpoint_path = model_checkpoint_path.replace(".ckpt", "")

    # Convert model to ONNX
    convert_model_to_onnx(
        model=new_model,
        model_checkpoint_path=model_checkpoint_path,
    )


def model_testing(
    datasets_path: str = DATASETS_PATH,
    data_split: str = "geo",
    band_list_dataset: str = ["B12", "B11", "B8A"],
    model_name: str = "unet_smp",
    models_path: str = MAIN_DIRECTORY,
    activation: str = "sigmoid",
    encoder: str = "mobilenet_v2",
    encoder_weights: str = "imagenet",
    n_filters: int = 32,
    loss_name: str = "focal_loss_smp",
    gamma: float = 2.0,
    alpha: float = 0.25,
    beta: float = 0.7,
    lr: float = DEFAULT_LR,
    lr_scheduler: int = 1,
    lr_finder: int = 1,
    weakly: int = 1,
    batch_size: int = DEFAULT_BS,
    device: str = DEVICE,
    n_cpu: int = N_CPU,
    seed: int = DEFAULT_SEED,
    precision: str = "16-mixed",
) -> None:
    """

    Attributes
    ----------

    datasets_path : str,
        Path to the directory of the training datasets.  Default=DATASETS_PATH

    data_split : str,
        help="Define the training split for the dataset. Default='geo', choices=['geo','random']

    band_list : str,
        Specify the band combination for the training dataset.  Default=["B12","B11","B8A"]

    model_name : str,
        Name of the desired segmentation model.  Default='unet_smp', choices=available_models

    models_path : str
        OPTIONAL: Path to the models folder. Default = MAIN_DIRECTORY

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
        OPTIONAL: Weight factor associated to class weight in Focal Loss 
        or to false negatives in Focal Tversky Loss. Default=0.25

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

    device : str,
        Device used for training the model. Options: 'cuda' or 'cpu. 
        Default=DEVICE, choices = ['cuda', 'cpu']

    n_cpu : int,
        Define the number of CPU cores to be used. Default is max number per device : N_CPU

    seed : int,
        Seed used for training the model. Default: 42. Default=DEFAULT_SEED

    precision : str,
        Specify whether mixed precision is used. Default: 16-mixed. 
        Default='16-mixed',choices=['16-mixed','32']

    Outputs
    -------
    Trained model in PyTorch and ONNX formats, and the training and testing logs

    Notes
    -----

    """

    pl.seed_everything(seed, workers=True)

    # To compute the loss during training the activation cannot be used, as the loss requires logits
    activation_training = None

    band_list_str = "_".join(band_list_dataset)

    if lr_scheduler:
        lr_scheduler_str = "_lr_scheduler"
    else:
        lr_scheduler_str = ""

    if weakly:
        weakly_str = "_weakly"
    else:
        weakly_str = ""

    if encoder_weights.lower() == "none":
        encoder_weights = None
        encoder_weights_str = ""
    else:
        encoder_weights_str = f"_{encoder_weights}"

    if encoder.lower() == "none":
        encoder = None
        encoder_str = ""
    else:
        encoder_str = f"_{encoder}"
    if n_filters < 10:
        n_filters_str = f"_0{n_filters}"
    else:
        n_filters_str = f"_{n_filters}"

    n_filters_str = f"_{n_filters}"

    if lr_finder:
        lr_str = "_lr_finder"
    else:
        lr_str = f"_lr{lr}"

    data_path = os.path.join(
        datasets_path, f"train_{data_split}_split{weakly_str}_{band_list_str}_dataset"
    )
    if os.path.isdir(data_path):

        filename_parts = [
            f"{model_name}{encoder_str}{encoder_weights_str}{n_filters_str}",
            f"fp{precision[:2]}",
            loss_name,
            f"gamma{gamma}",
            f"alpha{alpha}",
            f"{lr_str}{lr_scheduler_str}",
            f"{data_split}{weakly_str}",
            band_list_str,
        ]

        # Join only the non-empty parts with an underscore
        filename = "_".join(part for part in filename_parts if part)

        # Use the modern pathlib to create the full path
        model_name_path = os.path.join(MAIN_DIRECTORY, "models",filename)

    else:
        raise FileNotFoundError(
            f"The training dataset folder couldn't be found at: {data_path}"
        )

    os.makedirs(model_name_path, exist_ok=True)
    model_batch_size_path = os.path.join(
        model_name_path, f"batch_size_{batch_size}")
    model_seed_path = os.path.join(model_batch_size_path, f"seed_{seed}")

    # Obtains the respective run number, to avoid overwriting previous runs
    run_idx = sum(1 for file in os.listdir(
        model_seed_path) if file.startswith("run"))
    if run_idx > 0:
        run_idx -= 1

    model_main_path = os.path.join(model_seed_path, f"run_{run_idx}")

    logger = TensorBoardLogger(
        f"{model_main_path}/logs",
        name=f"{os.path.basename(model_seed_path)}",
        default_hp_metric=False,
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
        logging.WARNING)

    model = select_model(
        model_name=model_name,
        encoder=encoder,
        encoder_weights=encoder_weights,
        activation=activation_training,
        n_filters=n_filters,
    )

    loss = select_loss(
        loss_name=loss_name, gamma=gamma, alpha=alpha, beta=beta, weakly=weakly
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(model_main_path, "checkpoints"),
            filename=f"{os.path.basename(model_seed_path)}_" + "{epoch}",
            # filename=f"{os.path.basename(model_name_path)}",
            save_top_k=3,
            # every_n_epochs=10,
            monitor="val_loss",
            mode="min",
            verbose=False,
        ),
        EarlyStopping(
            monitor="val_loss", min_delta=1e-7, patience=8, verbose=False, mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if encoder and encoder_weights:  # Convert the images into the expected scale
        print(encoder, encoder_weights)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)
    else:
        preprocessing_fn = None

    model_checkpoint_path = [
        os.path.join(model_main_path, file)
        for file in os.listdir(model_main_path)
        if file.endswith(".ckpt")
    ][0]

    test_fn(
        model_main_path=model_main_path,
        checkpoint_path=model_checkpoint_path,
        model=model,
        loss=loss,
        callbacks=callbacks,
        logger=logger,
        images_path=data_path,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        batch_size=batch_size,
        device=device,
        num_workers=n_cpu,
        precision=precision,
        save_imgs=False,
    )

    # Create a new model by adding an additional sigmoid layer
    trained_model = SegTHRawSTrainModel.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path, model=model, loss_fn=loss
    )
    new_model = SegTHRawSModel(model=trained_model, activation=activation)

    # os.remove(model_checkpoint_path) #Delete previous checkpoints
    torch.save(new_model, model_checkpoint_path.replace(".ckpt", ""))
    model_checkpoint_path = model_checkpoint_path.replace(".ckpt", "")

    convert_model_to_onnx(
        model=new_model,
        model_checkpoint_path=model_checkpoint_path,
    )


def model_re_train_test(
    datasets_path: str = DATASETS_PATH,
    data_split: str = "geo",
    band_list_dataset: str = ["B12", "B11", "B8A"],
    model_name: str = "unet_smp",
    model_checkpoint_path: str = None,
    activation: str = "sigmoid",
    encoder: str = "mobilenet_v2",
    encoder_weights: str = "imagenet",
    n_filters: int = 32,
    loss_name: str = "focal_loss_smp",
    gamma: float = 2.0,
    alpha: float = 0.25,
    beta: float = 0.7,
    lr: float = DEFAULT_LR,
    lr_scheduler: int = 1,
    lr_finder: int = 1,
    weakly: int = 1,
    batch_size: int = DEFAULT_BS,
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    device: str = DEVICE,
    n_cpu: int = N_CPU,
    seed: int = DEFAULT_SEED,
    precision: str = "16-mixed",
) -> None:
    """

    Attributes
    ----------

    datasets_path : str,
        Path to the directory of the training datasets.  Default=DATASETS_PATH

    data_split : str,
        help="Define the training split for the dataset. Default='geo', choices=['geo','random']

    band_list : str,
        Specify the band combination for the training dataset.  Default=["B12","B11","B8A"]

    model_name : str,
        Name of the desired segmentation model.  Default='unet_smp', choices=available_models

    model_checkpoint_path : str,
        Path of the model's checkpoint to be retrained.

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
        OPTIONAL: Weight factor associated to class weight in Focal Loss
         or to false negatives in Focal Tversky Loss. Default=0.25

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
        Device used for training the model. Options: 'cuda' or 'cpu.
        Default=DEVICE, choices = ['cuda', 'cpu']

    n_cpu : int,
        Define the number of CPU cores to be used. Default is max number per device : N_CPU

    seed : int,
        Seed used for training the model. Default: 42. Default=DEFAULT_SEED

    precision : str,
        Specify whether mixed precision is used. Default: 16-mixed.
        Default='16-mixed',choices=['16-mixed','32']

    Outputs
    -------
    Trained model in PyTorch and ONNX formats, and the training and testing logs

    Notes
    -----

    """

    pl.seed_everything(seed, workers=True)

    # To compute the loss during training the activation cannot be used, as the loss requires logits
    activation_training = None

    band_list_str = "_".join(band_list_dataset)

    if lr_scheduler:
        lr_scheduler_str = "_lr_scheduler"
    else:
        lr_scheduler_str = ""

    if weakly:
        weakly_str = "_weakly"
    else:
        weakly_str = ""

    if encoder_weights.lower() == "none":
        encoder_weights = None
        encoder_weights_str = ""
    else:
        encoder_weights_str = f"_{encoder_weights}"

    if encoder.lower() == "none":
        encoder = None
        encoder_str = ""
    else:
        encoder_str = f"_{encoder}"

    n_filters_str = f"_{n_filters}"

    if lr_finder:
        lr_str = "_lr_finder"
    else:
        lr_str = f"_lr{lr}"

    data_path = os.path.join(
        datasets_path, f"train_{data_split}_split{weakly_str}_{band_list_str}_dataset"
    )
    if os.path.isdir(data_path):
        filename_parts = [
            f"{model_name}{encoder_str}{encoder_weights_str}{n_filters_str}",
            f"fp{precision[:2]}",
            loss_name,
            f"gamma{gamma}",
            f"alpha{alpha}",
            f"{lr_str}{lr_scheduler_str}",
            f"{data_split}{weakly_str}",
            band_list_str,
        ]

        # Join only the non-empty parts with an underscore
        filename = "_".join(part for part in filename_parts if part)

        # Use the modern pathlib to create the full path
        model_name_path = os.path.join(MAIN_DIRECTORY, "models",filename)
    else:
        raise FileNotFoundError(
            f"The training dataset folder couldn't be found at: {data_path}"
        )

    os.makedirs(model_name_path, exist_ok=True)
    model_batch_size_path = os.path.join(
        model_name_path, f"batch_size_{batch_size}")
    model_seed_path = os.path.join(model_batch_size_path, f"seed_{seed}")

    os.makedirs(model_batch_size_path, exist_ok=True)
    os.makedirs(model_seed_path, exist_ok=True)

    # Obtains the respective run number, to avoid overwriting previous runs
    run_idx = sum(1 for file in os.listdir(
        model_seed_path) if file.startswith("run"))

    model_main_path = os.path.join(model_seed_path, f"run_{run_idx}")
    os.makedirs(model_main_path, exist_ok=True)

    metrics_path = os.path.join(model_main_path, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    myriad_path = os.path.join(model_main_path, "MYRIAD")
    os.makedirs(myriad_path, exist_ok=True)

    # logger = CSVLogger(f"{model_main_path}/csv_logs", name=f"{os.path.basename(model_seed_path)}")

    logger = TensorBoardLogger(
        f"{model_main_path}/logs",
        name=f"{os.path.basename(model_seed_path)}",
        default_hp_metric=False,
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
        logging.WARNING)

    model = select_model(
        model_name=model_name,
        encoder=encoder,
        encoder_weights=encoder_weights,
        activation=activation_training,
        n_filters=n_filters,
    )

    loss = select_loss(
        loss_name=loss_name, gamma=gamma, alpha=alpha, beta=beta, weakly=weakly
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(model_main_path, "checkpoints"),
            filename=f"{os.path.basename(model_seed_path)}_" + "{epoch}",
            # filename=f"{os.path.basename(model_name_path)}",
            save_top_k=3,
            # every_n_epochs=10,
            monitor="val_loss",
            mode="min",
            verbose=False,
        ),
        EarlyStopping(
            monitor="val_loss", min_delta=1e-7, patience=8, verbose=False, mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if encoder and encoder_weights:  # Convert the images into the expected scale
        print(encoder, encoder_weights)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)
    else:
        preprocessing_fn = None

    last_epoch_saved = int(
        re.search(r"=(\d+)\.ckpt", model_checkpoint_path).group(1))

    trained_model = SegTHRawSTrainModel.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path,
        model=model,
        loss_fn=loss,
        last_saved_epoch=last_epoch_saved,
    )

    model_checkpoint_path = retrain_fn(
        callbacks=callbacks,
        model=model,
        loss_fn=loss,
        activation=activation_training,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        logger=logger,
        images_path=data_path,
        optim_dict=None,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_finder=lr_finder,
        weakly=weakly,
        batch_size=batch_size,
        device=device,
        num_workers=n_cpu,
        precision=precision,
        metrics_path=metrics_path,
        lightning_model=trained_model,
    )

    # Move the final model checkpoint to the main model directory to the
    shutil.copyfile(
        model_checkpoint_path,
        os.path.join(model_main_path, os.path.basename(model_checkpoint_path)),
    )
    shutil.rmtree(os.path.dirname(model_checkpoint_path))

    model_checkpoint_path = os.path.join(
        model_main_path, os.path.basename(model_checkpoint_path)
    )

    # Testing
    test_fn(
        model_main_path=model_main_path,
        checkpoint_path=model_checkpoint_path,
        model=model,
        loss=loss,
        callbacks=callbacks,
        logger=logger,
        images_path=data_path,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        batch_size=batch_size,
        device=device,
        num_workers=n_cpu,
        precision=precision,
    )

    model_checkpoint_path = [
        os.path.join(model_main_path, checkpoint_path)
        for checkpoint_path in os.listdir(model_main_path)
        if checkpoint_path.endswith(".ckpt")
    ][0]

    # Create the inference model by adding an additional sigmoid layer
    trained_model = SegTHRawSTrainModel.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path, model=model, loss_fn=loss
    )
    new_model = SegTHRawSModel(model=trained_model, activation=activation)

    # Save inference model
    torch.save(new_model, model_checkpoint_path.replace(".ckpt", ""))
    model_checkpoint_path = model_checkpoint_path.replace(".ckpt", "")

    convert_model_to_onnx(
        model=new_model,
        model_checkpoint_path=model_checkpoint_path,
    )
