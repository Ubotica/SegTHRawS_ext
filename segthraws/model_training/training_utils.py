"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning as pl
import albumentations as albu
import matplotlib.pyplot as plt
import torchvision.transforms as T

from typing import Any, Union
from torchmetrics import FBetaScore
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.utils import metrics

from ..utils import read_binary_image


from .train_constants import DEFAULT_BS, DEFAULT_LR, N_CPU, DEVICE, mean_imagenet_imgs, std_imagenet_imgs

class SegTHRawSDataset(Dataset):
    def __init__(self,
                 stage: str,
                 images_path: str,
                 augmentation: Union[T.Compose, albu.Compose] = None,
                 preprocessing: Union[T.Compose, albu.Compose] = None,
                 shuffle: bool = True,
                 ) -> None:
        """Create the Dataset object for training the model with Lightning.
        
        Attributes
        ----------

        stage : str
            Stage of the training process. Choices: train, val, and test.
        
        images_path : str
                Path of the folder where the images are located.
        
        augmentation : Union[T.Compose, albu.Compose]
            albumentations.Compose transformation that applies the image augmentations
        
        preprocessing : Union[T.Compose, albu.Compose]
            albumentations.Compose transformation for data with pre-trained weights 
        
        shuffle: bool 
            Random shuffle the images o. Default = True

        Outputs
        -------
        model_checkpoint_path: str
            Path to the checkpoint of the best trained model according to the validation loss metric
        
        Notes
        -----

        """
        self.stage = stage
        self.images_path = images_path
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.shuffle = shuffle

        self.total_len = None
        self._images, self._masks = self.__create_dataset()

    def __create_dataset(self) -> dict:
        dict_paths = {
            "image": [],
            "mask": []
        }

        images_path = os.path.join(self.images_path,self.stage,'images')

        images_path_shuffle = os.listdir(images_path)
            
        if self.shuffle:
            random.shuffle(images_path_shuffle)

        for image_name in images_path_shuffle:
            dict_paths["image"].append(os.path.join(images_path,image_name))
            dict_paths["mask"].append(os.path.join(os.path.dirname(images_path),'masks',image_name.replace('.bin','_mask.bin')))

        dataframe = pd.DataFrame(
            data=dict_paths,
            index=np.arange(0, len(dict_paths["image"]))
        )
        self.total_len = len(dataframe)
        data_dict = {self.stage: (dataframe["image"].values,dataframe["mask"].values)}

        return data_dict[self.stage]

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx) -> tuple:

        image = read_binary_image(image_path=self._images[idx],
                                  dtype=np.float32,
                                  shape=[256,256,3])

        mask = read_binary_image(image_path=self._masks[idx],
                                  dtype=np.float32,
                                  shape=[256,256,1])
        
        # # apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


def get_training_augmentation():
    """Construct augmentations transformation
    
    Attributes
    ----------

    None
    
    Outputs
    -------
    augmentations: albumentations.Compose
        Augmentations function 
    
    Notes
    -----
    
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.GridDistortion(distort_limit=0.2,p=0.5),
        # albu.ColorJitter(brightness=(0.75,1.25),contrast=(0.75,1.25),saturation=0,hue=0,p=0.5),
    ]
    return albu.Compose(train_transform)

def get_desired_format(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transformation.
    
    Attributes
    ----------
    preprocessing_fn : Any
        pre_processing normalization function, which substracts the mean and std of the input images and normalize.
    
    Outputs
    -------
    transform: albumentations.Compose
        Pre-processing transformation needed to normalize the input data to that of the pre-trained weights.
    
    Notes
    -----

    """
    if preprocessing_fn:
        
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=get_desired_format, mask=get_desired_format),
        ]
    else:
        _transform = [
            albu.Lambda(image=get_desired_format, mask=get_desired_format),
        ]
    return albu.Compose(_transform)

class SegTHRawSDataModule(pl.LightningDataModule):
    """DataModule that provides the dataset and dataloaders to the model.

    Attributes
    ----------
        images_path : str
            Path to the input images 
        augmentation : Union[T.Compose, albu.Compose]
            albumentations.Compose transformation that applies the image augmentations
        preprocessing : Union[T.Compose, albu.Compose]
            albumentations.Compose transformation for data with pre-trained weights 
        batch_size : int
            Batch size of the dataloaders 
        num_workers : int
            Number of CPU cores  

    Outputs
    -------
    DataModule : pl.LightningDataModule
        Datamodule object to pass to Lightning's trainer
    
    Notes
    -----

    """
    def __init__(self,images_path: str,
                 augmentation: Union[T.Compose, albu.Compose] = None,
                 preprocessing: Union[T.Compose, albu.Compose] = None,
                 batch_size: int = DEFAULT_BS,
                 num_workers: int = N_CPU,
                 ) -> pl.LightningDataModule:
        
        super().__init__()

        self.images_path = images_path
        self.train_augmentation = augmentation
        self.eval_augmentation = augmentation # At the moment, using the same as for training
        self.preprocessing = preprocessing    # Needed for pre-trained backbones in imagenet
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None


    def setup(self, stage: str = None) -> None:
        self.data_train = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.train_augmentation,
            preprocessing=self.preprocessing,
            stage="train",
            shuffle=True,
            )

        self.data_val = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="val",
            shuffle=True,
            )

        self.data_test = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="test",
            shuffle=True,
            )

        self.data_predict = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="test",
            shuffle=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    

class SegTHRawSTrainModel(pl.LightningModule):
    """Segmentation model used for training
        
    Attributes:
    ----------
    model : nn.Module
        Input segmentation model in PyTorch format
    loss_fn : Any
        Loss function
    model_folder_path : str
        Path for saving the testing output masks
    logger : Any
        Logger function used by Lightning to save the values
    optim_dict : dict
        Dictionary for optimizer an learning rate monitoring functions. Default = None
    activation : str
        Final layer Activation. Default = None
    batch_size : int
        Batch size hyperparameter. Default = DEFAULT_BS  
    lr : float
        Learning rate hyperparameter. Default = DEFAULT_LR  
    weakly : bool
        Indicate if weakly labelling is used, to modify the metrics. Default = True
    device : str
        Indicate if cuda is used. Default = 'cuda'
    num_classes : int
        Number of segmentation classes. Default = 2
    mean_imagenet_imgs : list
        Mean of the imagenet images required for saving testing output masks when pre-trained weights are used
    std_imagenet_imgs : list
        Dtandard deviation of the imagenet images required for saving testing output masks when pre-trained weights are used
    lr_scheduler : bool
        Indicate if the ReduceLROnPlateau learning rate scheduler is used. Default = True
    last_saved_epoch : int
        Last epoch saved. Used for re-training models, and not start counting from 0. Default: 0
    save_imgs : bool
        Determine if the testing output images want to be saved. Default = True

    Output
    ------
    segthraws_model : pl.LightningModule
        Trained segmentation model that obtains the logits. It is passed to Lightning's trainer
    
    Notes
    -----
    """
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Any,
                 model_folder_path: str,
                 logger: Any,
                 optim_dict: dict = None,
                 activation: str = None,
                 batch_size: int = DEFAULT_BS,
                 lr: float = DEFAULT_LR,
                 weakly: bool = True,
                 device: str = DEVICE,
                 num_classes: int = 2,
                 mean_imagenet_imgs: list = mean_imagenet_imgs,
                 std_imagenet_imgs: list = std_imagenet_imgs,
                 lr_scheduler: bool = True,
                 last_saved_epoch: int = 0,
                 save_imgs: bool = True):

        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn'])

        self.model = model

        self.criterion = loss_fn
        self.optim_dict = optim_dict
        self.batch_size = batch_size
        self._device = device
        self.activation = activation
        self.lr_scheduler = lr_scheduler
        self.weakly = weakly
        
        self.mean_imagenet_imgs = mean_imagenet_imgs
        self.std_imagenet_imgs = std_imagenet_imgs

        self.model_folder_path = model_folder_path

        logger.log_hyperparams(self.hparams)

        self.last_saved_epoch = last_saved_epoch
        self.save_imgs = save_imgs

        if weakly:
            self.num_classes = 2
            ignore_index = -1 
        else:
            self.num_classes = 1
            ignore_index = None

        self.step_outputs = {
            "loss"              : [],
            "fbeta"             : [],
            "fbeta_event"       : [],
            "IoU"               : [],
            "IoU_event"         : [],
            "IoU_event_0.5"     : [],
            "IoU_event_0.9"     : [],
            "IoU_notevent"      : [],
            "IoU_notevent_0.5"  : [],
            "IoU_notevent_0.9"  : [],
        }

        self.stage_outputs = {
            "train": self.step_outputs,
            "val": self.step_outputs,
            "test": self.step_outputs
        }

        self.metrics = {
            "fbeta_score": FBetaScore(task="binary",
                                      beta=1.0,
                                      threshold=0.5,
                                      num_classes=num_classes,
                                      average="micro",
                                      ignore_index=ignore_index,
                                      validate_args=True).to(self._device),
            "IoU": metrics.IoU(),
        }

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        x, y = x.to(self._device),y.to(self._device)

        logits = self.forward(x.to(torch.float32))
        
        if not self.activation:
            predictions = (torch.sigmoid(logits)>0.5).float()
        else:
            predictions = (logits>0.5).float()
        
        loss = self.criterion(logits, y)

        fbeta_score = self.metrics["fbeta_score"](predictions, y) #This includes ignore index inside the function

        if self.weakly:

            not_weakly_classes = y != -1
            predictions_global = predictions[not_weakly_classes]
            y_global = y[not_weakly_classes]


        IoU_score = self.metrics["IoU"](predictions_global, y_global)

        self.stage_outputs[stage]["loss"].append(loss)
        self.stage_outputs[stage]["fbeta"].append(fbeta_score)
        self.stage_outputs[stage]["IoU"].append(IoU_score)


        IoU_mask_event = [] #List that saves the IoU of those masks with events
        IoU_mask_notevent = [] #List that saves the IoU of those masks with events
        fbeta_events_list = []
        for idx,mask_batch in enumerate(y):
            if torch.any(mask_batch!=0):
                if self.weakly:
                    not_weakly_classes = mask_batch != -1
                    predictions_event = predictions[idx][not_weakly_classes]
                    mask_batch_event = mask_batch[not_weakly_classes]
                    IoU_mask_event.append(self.metrics["IoU"](predictions_event, mask_batch_event))
                    fbeta_events_list.append(self.metrics["fbeta_score"](predictions_event, mask_batch_event))
                else:
                    IoU_mask_event.append(self.metrics["IoU"](predictions[idx], mask_batch))
                    fbeta_events_list.append(self.metrics["fbeta_score"](predictions[idx], mask_batch))
            else:
                if self.weakly:
                    not_weakly_classes = mask_batch != -1
                    predictions_notevent = predictions[idx][not_weakly_classes]
                    mask_batch_notevent = mask_batch[not_weakly_classes]
                    IoU_mask_notevent.append(self.metrics["IoU"](predictions_notevent, mask_batch_notevent))
                else:
                    IoU_mask_notevent.append(self.metrics["IoU"](predictions[idx], mask_batch))


        if len(IoU_mask_event)>0:        

            IoU_mask_event    = torch.tensor(IoU_mask_event)
            fbeta_events_list = torch.tensor(fbeta_events_list)

            IoU_events    = torch.mean(IoU_mask_event)
            fbeta_events  = torch.mean(fbeta_events_list)

            self.stage_outputs[stage]["fbeta_event"].append(fbeta_events)
            self.stage_outputs[stage]["IoU_event"].append(IoU_events)

            if torch.any(IoU_mask_event>0.5):
                IoU_0_5_events = torch.mean(torch.tensor([item for item in IoU_mask_event if item>0.5]))
                self.stage_outputs[stage]["IoU_event_0.5"].append(IoU_0_5_events)

                if torch.any(IoU_mask_event>0.9):
                    IoU_0_9_events = torch.mean(torch.tensor([item for item in IoU_mask_event if item>0.9]))
                    self.stage_outputs[stage]["IoU_event_0.9"].append(IoU_0_9_events)


        if len(IoU_mask_notevent)>0:        
            IoU_mask_notevent = torch.tensor(IoU_mask_notevent)
            IoU_notevents = torch.mean(IoU_mask_notevent)
            self.stage_outputs[stage]["IoU_notevent"].append(IoU_notevents)


       
            if torch.any(IoU_mask_notevent>0.5):
                IoU_0_5_notevents = torch.mean(torch.tensor([item for item in IoU_mask_notevent if item>0.5]))
                self.stage_outputs[stage]["IoU_notevent_0.5"].append(IoU_0_5_notevents)

                if torch.any(IoU_mask_notevent>0.9):
                    IoU_0_9_notevents = torch.mean(torch.tensor([item for item in IoU_mask_notevent if item>0.9])) 
                    self.stage_outputs[stage]["IoU_notevent_0.9"].append(IoU_0_9_notevents)


        fbeta_score = self.metrics["fbeta_score"](predictions, y) #This includes ignore index inside the function

        if stage=='test' and self.save_imgs:
            test_masks_path = os.path.join(self.model_folder_path,'test_masks_comparison')

            if os.path.isdir(test_masks_path):
                test_idx = len(os.listdir(test_masks_path))
            else:
                test_idx = 0

            os.makedirs(test_masks_path,exist_ok=True)
            for batch_idx in range(x.shape[0]):
                fig, ax = plt.subplots(1,3,figsize = (9,3))
                
                image = np.transpose(x[batch_idx].cpu().detach().numpy(),(1,2,0))
                mask = np.transpose(y[batch_idx].cpu().detach().numpy(),(1,2,0))
                prediction_mask = np.transpose(predictions[batch_idx].cpu().detach().numpy(),(1,2,0))

                #These conditions are included to avoid clipping warnings
                image[image<0]=0 
                mask[mask==-1]=0.5

                plt.suptitle(f' Test masks comparison {test_idx+batch_idx}', fontsize=14)
                ax[0].imshow(image)
                ax[1].imshow(mask)
                ax[2].imshow(prediction_mask)
                plt.tight_layout()
                plt.savefig(os.path.join(test_masks_path,f'test_mask_comparison_{test_idx+batch_idx}'))
                plt.close()

        return loss

    def shared_epoch_end(self, stage: Any):
        loss = torch.mean(torch.tensor([
            loss for loss in self.stage_outputs[stage]["loss"]
        ]))
        
        fbeta = torch.mean(torch.tensor(
            [fbeta_score for fbeta_score in self.stage_outputs[stage]["fbeta"]
             ]))
        
        fbeta_event = torch.mean(torch.tensor(
            [fbeta_score for fbeta_score in self.stage_outputs[stage]["fbeta_event"]
             ]))

        IoU = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU"]
                 ]))
        
        IoU_event = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_event"]
                 ]))
        
        IoU_event_0_5 = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_event_0.5"]
                 ]))
      
        IoU_event_0_9 = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_event_0.9"]
                 ]))

        IoU_notevent = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_notevent"]
                 ]))
        
        IoU_notevent_0_5 = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_notevent_0.5"]
                 ]))
        
        IoU_notevent_0_9 = torch.mean(torch.tensor(
                [IoU_score for IoU_score in self.stage_outputs[stage]["IoU_notevent_0.9"]
                 ]))


        metrics = {
            f"{stage}_loss"              : loss,
            f"{stage}_fbeta"             : fbeta,
            f"{stage}_fbeta_event"       : fbeta_event,
            f"{stage}_IoU"               : IoU,
            f"{stage}_IoU_event"         : IoU_event,
            f"{stage}_IoU_event_0.5"     : IoU_event_0_5,
            f"{stage}_IoU_event_0.9"     : IoU_event_0_9,
            f"{stage}_IoU_notevent"      : IoU_notevent,
            f"{stage}_IoU_notevent_0.5"  : IoU_notevent_0_5,
            f"{stage}_IoU_notevent_0.9"  : IoU_notevent_0_9,
            "step"                       : self.current_epoch+self.last_saved_epoch,
        }
        self.log_dict(metrics, prog_bar=True,on_epoch=True)

    def training_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="val")

    def test_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="test")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="val")

    def on_test_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="test")



    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        logits = self.forward(x.to(torch.float32))
        
        
        if not self.activation:
            predictions = (torch.sigmoid(logits)>0.5).float()
        else:
            predictions = (logits>0.5).float
        

        image = np.transpose(x[batch_idx].cpu().detach().numpy(),(1,2,0))
        mask = np.transpose(y[batch_idx].cpu().detach().numpy(),(1,2,0))
        prediction_mask = np.transpose(predictions[batch_idx].cpu().detach().numpy(),(1,2,0))

        image = image*np.array(self.std_imagenet_imgs) + np.array(self.mean_imagenet_imgs)


        return image,mask,prediction_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr
        )


        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=5
            ),
            # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer=optimizer,
            #     T_max=2,
            #     eta_min=0.0009
            # ),
            "interval": "epoch",
            "monitor": "val_loss"
        }
        
        optimization_dictionary = {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return optimization_dictionary


class SegTHRawSModel(pl.LightningModule):
    """Segmentation model used for inference. It adds an addiitonal sigmoid layer
    
    Attributes
    ----------
    model : pl.LightningModule
        Trained model
    device : str
        Indicate if cuda is used. Default = 'cuda'
    activation : str
        Final layer Activation. Default = 'sigmoid'

    Outputs
    -------
    segthraws_model : pl.LightningModule 
        Segmentation model that obtains the final output masks
    
    Notes
    -----

    """
    def __init__(self,
                 model: pl.LightningModule,
                 device: str = DEVICE,
                 activation: str = 'sigmoid'):
        
        super().__init__()

        self.model = model
        self.metrics  = model.metrics
        
        self._device = device

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()

        else:
            raise ValueError(
                f"Activation should be sigmoid"
                f"; got {activation}"
            )

    def forward(self, x):

        self.model.eval()
        with torch.no_grad():
            x = self.model(x)

            output = (self.activation(x)>0.5).float()

        return output


    def test_step(self, batch: Any, batch_idx: Any):

        x, y = batch
        x, y = x.to(self._device),y.to(self._device)

        predictions = self.forward(x.to(torch.float32))
        
        stage = 'test'
        fbeta_score = self.metrics["fbeta_score"](predictions, y)
        IoU_score = self.metrics["IoU"](predictions, y)

        self.log(f'{stage}_fbeta'  , fbeta_score   , prog_bar=True , on_step=False , on_epoch=True)
        self.log(f'{stage}_IoU'    , IoU_score     , prog_bar=True , on_step=False , on_epoch=True)

        return predictions

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        predictions = self.forward(x.to(torch.float32))
        return predictions
