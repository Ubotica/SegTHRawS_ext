import os
import lightning as pl

from typing import Any, Union
from lightning.pytorch.tuner import Tuner

from training_utils import SegTHRawSDataModule, SegTHRawSTrainModel
from train_constants import N_CPU, DEVICE
from train_constants import DEFAULT_SEED, DEFAULT_BS, DEFAULT_LR 
from train_constants import DEFAULT_MIN_EPOCHS, DEFAULT_MAX_EPOCHS

def main_train(callbacks: list,
         model: Union[list, tuple],
         loss_fn: Any,
         ACTIVATION: str,
         augmentation: Any,
         preprocessing: Any,
         logger: Any,
         images_path: str,
         optim_dict: dict = None,
         min_epochs: int  = DEFAULT_MIN_EPOCHS,
         max_epochs: int = DEFAULT_MAX_EPOCHS,
         lr: float = DEFAULT_LR,
         lr_scheduler: bool = True,
         weakly: bool = True,
         batch_size: int = DEFAULT_BS,
         precision: str = '16-mixed',
         metrics_path: str = None,
         seed: int = DEFAULT_SEED,
         device: str = DEVICE,
         num_workers: int = N_CPU,
         ) -> str:

    # torch.manual_seed(seed)

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
        precision=precision
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

    # LightningModule
    lightning_model = SegTHRawSTrainModel(
        model=model,
        loss_fn=loss_fn,
        ACTIVATION=ACTIVATION,
        optim_dict=optim_dict,
        batch_size=batch_size,
        lr=lr,
        lr_scheduler=lr_scheduler,
        weakly=weakly,
        seed=seed,
        model_folder_path=os.path.dirname(metrics_path)
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(lightning_model,datamodule=datamodule)

    lr_fig = lr_finder.plot(suggest=True)
    lr_fig.savefig(os.path.join(metrics_path,'lr_finder'))
    lightning_model.hparams.lr = lr_finder.suggestion()

    # Start training
    trainer.fit(model=lightning_model, datamodule=datamodule)

    return callbacks[0].best_model_path