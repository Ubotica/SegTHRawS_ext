

import os
import re
import torch
import shutil
import torch.nn as nn
import albumentations as albu
import pytorch_lightning as pl
import torchvision.transforms as T

from typing import Any, Union
from torch.nn.modules.loss import _Loss

from .train_constants import DEFAULT_SEED, DEFAULT_BS, DEVICE, N_CPU
from .training_utils import SegTHRawSTrainModel, SegTHRawSDataModule


def main_test(model_main_path: str,
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


    checkpoints_paths = [os.path.join(model_main_path,'checkpoints',checkpoint_path) for checkpoint_path in os.listdir(os.path.join(model_main_path,'checkpoints')) if checkpoint_path[-5:]=='.ckpt']
    checkpoint_path = max(checkpoints_paths, key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))
    
    
    shutil.copyfile(checkpoint_path,os.path.join(model_main_path,os.path.basename(checkpoint_path)))
    shutil.rmtree(os.path.join(model_main_path,'checkpoints'))

    checkpoint_path = os.path.join(model_main_path,os.path.basename(checkpoint_path))


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

    trainer.test(model=trained_model,datamodule=datamodule)[0];
