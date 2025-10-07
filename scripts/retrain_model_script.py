"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import sys
import argparse

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from segthraws.model_training.training_scripts import model_re_train_test

from segthraws.model_training.train_constants import (
    DATASETS_PATH,
    AVAILABLE_MODELS,
    AVAILABLE_ENCODERS,
    AVAILABLE_LOSSES,
)
from segthraws.model_training.train_constants import (
    DEFAULT_BS,
    DEFAULT_LR,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_SEED,
)
from segthraws.model_training.train_constants import DEVICE, N_CPU

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SegTHRawS model training")

    parser.add_argument(
        "--datasets_path",
        type=str,
        help="Path to the directory of the training datasets",
        default=DATASETS_PATH,
    )

    parser.add_argument(
        "--data_split",
        type=str,
        help="Define the training split for the dataset. Options: 'geo' or 'random'",
        default="geo",
        choices=["geo", "random"],
    )

    parser.add_argument(
        "--band_list",
        type=str,
        help='Specify the band combination for the training dataset. Example: ["B12","B11","B8A"] ',
        nargs="+",
        default=["B12", "B11", "B8A"],
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the desired segmentation model",
        default="unet_smp",
        choices=AVAILABLE_MODELS,
    )

    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        help="Path to the checkpoint of the model to be retrained",
        default=None,
    )

    parser.add_argument(
        "--activation",
        type=str,
        help="Final activation layer of the model",
        default="sigmoid",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Name of the desired encoder",
        default="mobilenet_v2",
        choices=AVAILABLE_ENCODERS,
    )

    parser.add_argument(
        "--encoder_weights",
        type=str,
        help="Name of the pre-trained weights of the encoder",
        default="imagenet",
    )

    parser.add_argument(
        "--n_filters",
        type=int,
        help="Number of filters for the U-Net modification",
        default=32,
    )

    parser.add_argument(
        "--loss",
        type=str,
        help="Name of the desired loss function",
        default="focal_loss_smp",
        choices=AVAILABLE_LOSSES,
    )

    parser.add_argument(
        "--gamma", type=float, help="Importance factor for the focal loss", default=2.0
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="OPTIONAL: Weight factor associated to class weight in Focal Loss or to false negatives in Focal Tversky Loss",
        default=0.25,
    )

    parser.add_argument(
        "--beta",
        type=float,
        help="OPTIONAL: Weight factor associated to false positives for Focal Tversky Loss",
        default=0.7,
    )

    parser.add_argument(
        "--lr", type=float, help="Desired initial learning rate", default=DEFAULT_LR
    )

    parser.add_argument(
        "--lr_scheduler",
        type=int,
        help="Decide if a decreasing lr scheduler is used. Options: 0 or 1. Default: 1",
        default=1,
        choices=[0, 1],
    )

    parser.add_argument(
        "--lr_finder",
        type=int,
        help="Select if a learning finder is used. Options: 0 or 1. Default: 1",
        default=1,
        choices=[0, 1],
    )

    parser.add_argument(
        "--weakly",
        type=int,
        help="Decide if weakly segmentation is used. Options: 0 or 1. Default: 1",
        default=1,
        choices=[0, 1],
    )

    parser.add_argument("--bs", type=int, help="Desired batch size", default=DEFAULT_BS)

    parser.add_argument(
        "--min_epochs",
        type=int,
        help="Number of minimum epochs for training the model",
        default=DEFAULT_MIN_EPOCHS,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of maximum epochs for training the model",
        default=DEFAULT_MAX_EPOCHS,
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device used for training the model. Options: 'cuda' or 'cpu'",
        default=DEVICE,
    )

    parser.add_argument(
        "--n_cpu",
        type=int,
        help="Define the number of CPU cores to be used. Default is max number per device",
        default=N_CPU,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed used for training the model. Default: 42",
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="Specify whether mixed precision is used. Default: 16-mixed",
        default="16-mixed",
        choices=["16-mixed", "32"],
    )

    # parse the arguments
    args = parser.parse_args()

    datasets_path = args.datasets_path
    data_split = args.data_split
    band_list_dataset = args.band_list
    model_name = args.model_name
    model_checkpoint_path = args.model_ckpt_path
    activation = args.activation
    encoder = args.encoder
    encoder_weights = args.encoder_weights
    n_filters = args.n_filters
    loss_name = args.loss
    gamma = args.gamma
    alpha = args.alpha
    beta = args.beta
    lr = args.lr
    lr_scheduler = bool(args.lr_scheduler)
    lr_finder = bool(args.lr_finder)
    batch_size = args.bs
    min_epochs = args.min_epochs
    max_epochs = args.max_epochs
    device = args.device
    n_cpu = args.n_cpu
    seed = args.seed
    precision = args.precision
    weakly = bool(args.weakly)

    model_re_train_test(
        datasets_path=datasets_path,
        data_split=data_split,
        band_list_dataset=band_list_dataset,
        model_name=model_name,
        model_checkpoint_path=model_checkpoint_path,
        activation=activation,
        encoder=encoder,
        encoder_weights=encoder_weights,
        n_filters=n_filters,
        loss_name=loss_name,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_finder=lr_finder,
        weakly=weakly,
        batch_size=batch_size,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        device=device,
        n_cpu=n_cpu,
        seed=seed,
        precision=precision,
    )
