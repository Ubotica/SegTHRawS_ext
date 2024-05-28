

import torch
import torch.nn as nn
import lightning as pl
import albumentations as albu
import torchvision.transforms as T

from typing import Any, Union
from training_utils import SegTHRawSTrainModel, SegTHRawSDataModule
from torch.nn.modules.loss import _Loss

from train_constants import DEFAULT_SEED, DEFAULT_BS, DEVICE, N_CPU


def main_test(model_main_path: str,
              checkpoint_path: str,
              model: nn.Module,
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
              seed : int = DEFAULT_SEED) -> None:

    torch.manual_seed(seed)

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
        seed=seed
    )

    trainer.predict(model=trained_model,datamodule=datamodule);