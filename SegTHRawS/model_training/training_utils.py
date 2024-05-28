import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning as pl
import albumentations as albu
import torchvision.transforms as T

import matplotlib.pyplot as plt

from typing import Any, Union
from torchmetrics import FBetaScore
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.utils import metrics

import sys
sys.path.insert(1,"..")
from utils import read_binary_image

# from torchmetrics import Accuracy, JaccardIndex

from train_constants import DEFAULT_SEED, DEFAULT_BS, DEFAULT_LR, N_CPU, DEVICE, mean_imagenet_imgs, std_imagenet_imgs

class SegTHRawSDataset(Dataset):
    def __init__(self,
                 stage: str,
                 images_path: str,
                 augmentation: Any = None,
                 preprocessing: Any = None,
                 shuffle: bool = True,
                 seed: int = DEFAULT_SEED) -> None:

        self.__attribute_checking(images_path,
                                  stage, shuffle)

        self.images_path = images_path

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.stage = stage
        self.shuffle = shuffle
        self.seed = seed
        self.total_len = None
        self._images, self._masks = self.__create_dataset()

        # torch.manual_seed(seed)
        # random.seed(seed)

    @staticmethod
    def __type_checking(images_path: str,
                        stage: str, shuffle: bool) -> None:
        assert isinstance(images_path, str)
        assert isinstance(stage, str)
        assert isinstance(shuffle, bool)



    @staticmethod
    def __path_checking(images_path: str) -> None:
        assert os.path.isdir(images_path)

    @staticmethod
    def __stage_checking(stage: str) -> None:
        assert stage in ["train", "test", "val"]

    @classmethod
    def __attribute_checking(cls, images_path: str,
                             stage: str,
                             shuffle: bool) -> None:

        cls.__type_checking(images_path=images_path,
                            stage=stage,
                            shuffle=shuffle)

        cls.__path_checking(images_path=images_path)

        cls.__stage_checking(stage=stage)

    def __create_dataset(self) -> dict:
        dict_paths = {
            "image": [],
            "mask": []
        }

        images_path = self.__split_data(self.stage)

        #### NEED TO ADD SHUFFLE ON HOW THE IMAGES ARE ACCESSED, AND INCLUDE SEED

        images_path_shuffle = os.listdir(images_path)
        
        random.shuffle(images_path_shuffle)

        for image_name in images_path_shuffle:
            dict_paths["image"].append(os.path.join(images_path,image_name))
            dict_paths["mask"].append(os.path.join(os.path.dirname(images_path),'masks',image_name.replace('_NIR_SWIR','_mask')))

        dataframe = pd.DataFrame(
            data=dict_paths,
            index=np.arange(0, len(dict_paths["image"]))
        )
        self.total_len = len(dataframe)
        data_dict = {self.stage: (dataframe["image"].values,dataframe["mask"].values)}

        return data_dict[self.stage]
    
    def __split_data(self, stage: str) -> str:
        return os.path.join(self.images_path,stage,'images')
    
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

#### NEED to create a class for the augmentations

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.GridDistortion(distort_limit=0.2,p=0.5),
        # albu.ColorJitter(brightness=(0.75,1.25),contrast=(0.75,1.25),saturation=0,hue=0,p=0.5),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class SegTHRawSDataModule(pl.LightningDataModule):
    def __init__(self,images_path: str,
                 augmentation: Union[T.Compose, albu.Compose] = None,
                 preprocessing: Any = None,
                 batch_size: int = DEFAULT_BS,
                 num_workers: int = N_CPU,
                 seed: int = DEFAULT_SEED):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.images_path = images_path
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None
        self.seed = seed
        self.train_augmentation = augmentation
        self.eval_augmentation = augmentation # At the moment, using the same as for training
        self.preprocessing = preprocessing    # Needed for pre-trained backbones in imagenet

        #CHECK IF THIS IS IMPLEMENTED CORRECTLY 
        # torch.manual_seed(seed)
        # random.seed(seed)

    def setup(self, stage: str = None) -> None:
        self.data_train = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.train_augmentation,
            preprocessing=self.preprocessing,
            stage="train",
            shuffle=True,
            seed=self.seed
            )

        self.data_val = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="val",
            shuffle=True,
            seed=self.seed
            )

        self.data_test = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="test",
            shuffle=True,
            seed=self.seed
            )

        self.data_predict = SegTHRawSDataset(
            images_path=self.images_path,
            augmentation=self.eval_augmentation,
            preprocessing=self.preprocessing,
            stage="test",
            shuffle=True,
            seed=self.seed
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
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Any,
                 model_folder_path: str,
                 optim_dict: dict = None,
                 ACTIVATION: str = None,
                 batch_size: int = DEFAULT_BS,
                 lr: float = DEFAULT_LR,
                 weakly: bool = True,
                 device: str = DEVICE,
                 num_classes: int = 2,
                 mean_imagenet_imgs: list = mean_imagenet_imgs,
                 std_imagenet_imgs: list = std_imagenet_imgs,
                 lr_scheduler: bool = True,
                 seed: int = DEFAULT_SEED):
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss_fn'])

        self.model = model

        self.criterion = loss_fn
        self.optim_dict = optim_dict
        self.batch_size = batch_size
        self._device = device
        self.ACTIVATION = ACTIVATION
        self.lr_scheduler = lr_scheduler
        self.weakly = weakly

        self.mean_imagenet_imgs = mean_imagenet_imgs
        self.std_imagenet_imgs = std_imagenet_imgs

        self.model_folder_path = model_folder_path

        if weakly:
            self.num_classes = 2
            ignore_index = -1 
        else:
            self.num_classes = 1
            ignore_index = None

        # torch.manual_seed(seed)
        # random.seed(seed)

        self.metrics = {
            #####Accuracy is removed because it is always higher than 0.99
            
            # "accuracy": Accuracy(task="binary",
            #                      threshold=0.5,
            #                      num_classes=num_classes,
            #                      validate_args=True,
            #                      ignore_index=None,
            #                      average="micro").to(self._device),

            ##### Jaccard is the same as IoU but does not handle when the output masks are empty
            
            # "jaccard_index": JaccardIndex(task="binary",
            #                               threshold=0.5,
            #                               num_classes=num_classes,
            #                               validate_args=True,
            #                               ignore_index=None,
            #                               average="macro").to(self._device),

            "fbeta_score": FBetaScore(task="binary",
                                      beta=1.0,
                                      threshold=0.5,
                                      num_classes=num_classes,
                                      average="micro",
                                      ignore_index=ignore_index,
                                      validate_args=True).to(self._device),

            # "fbeta_score": metrics.Fscore(),
            "IoU": metrics.IoU(),
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        x, y = x.to(self._device),y.to(self._device)

        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3 
        assert y.ndim == 4
        # assert y.max() <= 1 and y.min() >= 0

        logits = self.forward(x.to(torch.float32))
        
        if not self.ACTIVATION:
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

        self.log(f'{stage}_loss'       , loss          , prog_bar=True , on_step=False , on_epoch=True)
        self.log(f'{stage}_IoU_old'    , IoU_score     , prog_bar=True , on_step=False , on_epoch=True)
        self.log(f'{stage}_FBeta_old'  , fbeta_score   , prog_bar=True , on_step=False , on_epoch=True)


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

            self.log(f'{stage}_IoU_event'   , IoU_events    , prog_bar=True , on_step=False , on_epoch=True)
            self.log(f'{stage}_fbeta_event' , fbeta_events  , prog_bar=True , on_step=False , on_epoch=True)

            if torch.any(IoU_mask_event>0.5):
                IoU_0_5_events = torch.mean(torch.tensor([item for item in IoU_mask_event if item>0.5]))

                self.log(f'{stage}_IoU_event_0.5', IoU_0_5_events, prog_bar=True , on_step=False , on_epoch=True)

                if torch.any(IoU_mask_event>0.75):
                    IoU_0_75_events = torch.mean(torch.tensor([item for item in IoU_mask_event if item>0.75]))
                    self.log(f'{stage}_IoU_event_0.75', IoU_0_75_events, prog_bar=True , on_step=False , on_epoch=True)

                    if torch.any(IoU_mask_event>0.9):
                        IoU_0_9_events = torch.mean(torch.tensor([item for item in IoU_mask_event if item>0.9]))
                        self.log(f'{stage}_IoU_event_0.9', IoU_0_9_events, prog_bar=True , on_step=False , on_epoch=True)


        if len(IoU_mask_notevent)>0:        
            IoU_mask_notevent = torch.tensor(IoU_mask_notevent)
            IoU_notevents = torch.mean(IoU_mask_notevent)
            self.log(f'{stage}_IoU_notevent', IoU_notevents , prog_bar=True , on_step=False , on_epoch=True)


       
            if torch.any(IoU_mask_notevent>0.5):
                IoU_0_5_notevents = torch.mean(torch.tensor([item for item in IoU_mask_notevent if item>0.5]))
                self.log(f'{stage}_IoU_notevent_0.5',IoU_0_5_notevents, prog_bar=True , on_step=False , on_epoch=True)

                if torch.any(IoU_mask_notevent>0.75):
                    IoU_0_75_notevents = torch.mean(torch.tensor([item for item in IoU_mask_notevent if item>0.75]))
                    self.log(f'{stage}_IoU_notevent_0.75',IoU_0_75_notevents, prog_bar=True , on_step=False , on_epoch=True)

                    if torch.any(IoU_mask_notevent>0.9):
                        IoU_0_9_notevents = torch.mean(torch.tensor([item for item in IoU_mask_notevent if item>0.9]))
                        self.log(f'{stage}_IoU_notevent_0.9',IoU_0_9_notevents, prog_bar=True , on_step=False , on_epoch=True)            


        fbeta_score = self.metrics["fbeta_score"](predictions, y) #This includes ignore index inside the function

        if stage=='test':
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

                image = image*np.array(self.std_imagenet_imgs) + np.array(self.mean_imagenet_imgs)
                
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

    # def shared_epoch_end(self, stage: Any):

    def training_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="val")

    def test_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="test")

    # def on_train_epoch_end(self) -> None:
    #     return self.shared_epoch_end(stage="train")

    # def on_validation_epoch_end(self) -> None:
    #     return self.shared_epoch_end(stage="val")

    # def on_test_epoch_end(self) -> None:
    #     return self.shared_epoch_end(stage="test")



    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3
        assert y.ndim == 4
        # assert y.max() <= 1 and y.min() >= 0

        logits = self.forward(x.to(torch.float32))
        
        
        if not self.ACTIVATION:
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
        if self.lr_scheduler:
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=15
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
        else:
            optimization_dictionary = {"optimizer": optimizer}

        return self.optim_dict if self.optim_dict else optimization_dictionary


class SegTHRawSModel(pl.LightningModule):
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

############### REST OF ACTIVATION FUNCTIONS NOT IMPLEMENTED YET ###############
    # def Activation(self, activation, **params):
    #     if activation == "sigmoid":
    #         self.activation = nn.Sigmoid()

        # elif activation == "softmax2d":
        #     self.activation = nn.Softmax(dim=1, **params)
        # elif activation == "softmax":
        #     self.activation = nn.Softmax(**params)
        # elif activation == "logsoftmax":
        #     self.activation = nn.LogSoftmax(**params)
        # elif activation == "tanh":
        #     self.activation = nn.Tanh()
        # else:
        #     raise ValueError(
        #         f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh"
        #         f"/None; got {activation}"
        #     )

    def forward(self, x):

        self.model.eval()
        with torch.no_grad():
            x = self.model(x)

            output = (self.activation(x)>0.5).float()

        return output


    def test_step(self, batch: Any, batch_idx: Any):

        x, y = batch
        x, y = x.to(self._device),y.to(self._device)

        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3 
        assert y.ndim == 4
        # assert y.max() <= 1 and y.min() >= 0

        predictions = self.forward(x.to(torch.float32))
        
        stage = 'test'

        # accuracy = self.metrics["accuracy"](predictions, y)
        fbeta_score = self.metrics["fbeta_score"](predictions, y)
        IoU_score = self.metrics["IoU"](predictions, y)

        # self.log(f'{stage}_acc'    , accuracy      , prog_bar=True , on_step=False , on_epoch=True)
        self.log(f'{stage}_fbeta'  , fbeta_score   , prog_bar=True , on_step=False , on_epoch=True)
        self.log(f'{stage}_IoU'    , IoU_score     , prog_bar=True , on_step=False , on_epoch=True)


        # test_masks_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'test_masks_comparison')
        # os.makedirs(test_masks_path,exist_ok=True)
        # for batch_idx in range(self.batch_size):
        #     fig, ax = plt.subplots(1,3,figsize = (9,3))
        #     image = np.transpose(x[batch_idx]*self.std_imagenet + self.mean_imagenet,(1,2,0))
        #     mask = np.transpose(y[batch_idx],(1,2,0))
        #     prediction_mask = np.transpose(predictions[batch_idx],(1,2,0))

        #     plt.suptitle(f' Test masks comparison {self.test_idx+batch_idx}', fontsize=14)
        #     ax[0].imshow(image)
        #     ax[1].imshow(mask)
        #     ax[2].imshow(prediction_mask)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(test_masks_path,f'test_mask_comparison_{self.test_idx+batch_idx}'))
        # self.test_idx +=1



        return predictions

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3
        assert y.ndim == 4
        # assert y.max() <= 1 and y.min() >= 0

        predictions = self.forward(x.to(torch.float32))

        return predictions