"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""
import os
import re
import pickle
import random
import argparse
import numpy as np

from constants import DATASET_PATH, band_combinations_dict

def create_dataset_folders(new_dataset_path):
    
    for stage in ['train','val','test']:
        for files in ['images','masks']:
            os.makedirs(os.path.join(new_dataset_path,stage,files),exist_ok=True)


def split_events(new_dataset_path: str,
                 dataset_path: str = DATASET_PATH,
                 weakly: bool = False,
                 input_band_combination: list = ["B12",'B11',"B8A"],
                 train_split_ratio: float = 0.8,
                 val_split_ratio: float = 0.1,
                 test_split_ratio: float = 0.1,
                 seed: int = 42
                 ):

    random.seed(seed)
    if train_split_ratio+val_split_ratio+test_split_ratio != 1:
        if not test_split_ratio:
            test_split_ratio = 1 - (train_split_ratio + val_split_ratio)
        
        if train_split_ratio and not val_split_ratio and not test_split_ratio:
            print('Validation and test split ratios were not defined. Assuming equal split')
            val_split_ratio = (1-(train_split_ratio))/2
            test_split_ratio = val_split_ratio


    masks_path = os.path.join(dataset_path,'masks','weakly_segmentation')
    # events_path = os.path.join(dataset_path,'images','event','NIR_SWIR')


    if not new_dataset_path:
        new_dataset_path = os.path.join(os.path.dirname(dataset_path),'train_random_split_dataset')
        create_dataset_folders(new_dataset_path)

    train_max_idx = np.round(len(os.listdir(masks_path))*train_split_ratio).astype(int)

    val_max_idx = train_max_idx + np.round(len(os.listdir(masks_path))*val_split_ratio).astype(int)

    test_max_idx = len(os.listdir(masks_path))

    masks_paths_shuffle = os.listdir(masks_path)
    random.shuffle(masks_paths_shuffle)

    train_idx = 0
    val_idx = 0
    test_idx = 0

    for idx,mask_name in enumerate(masks_paths_shuffle):
        # patch_name = mask_name.replace('mask_weakly','NIR_SWIR')
        
        final_image = []
        for band in input_band_combination:
            for band_name,band_values in zip(band_combinations_dict.keys(),band_combinations_dict.values()):
                if band in band_values:

                    patch_name = mask_name.replace('mask_weakly',band_name)
                    image_path = os.path.join(dataset_path,'images','event',band_name,patch_name)
                    with open(image_path,'rb') as image_file:
                        image = pickle.load(image_file)

                    final_image.append(image[:,:,band_values.index(band)])
                    break
            
        image = np.transpose(np.array(final_image),(1,2,0))

        # with open(os.path.join(events_path,patch_name),'rb') as image_file:
        #     image = pickle.load(image_file)

        with open(os.path.join(masks_path,mask_name),'rb') as mask_file:
            mask = pickle.load(mask_file)

        if not weakly: #For supervised binary segmentation
            mask[mask == -1] = 0

        patch_name = patch_name.replace(f'_{band_name}',"")

        if idx< train_max_idx:

            with open(os.path.join(new_dataset_path,'train','masks',mask_name.replace('_mask_weakly.pkl','_mask.bin')),'wb') as file:
                mask.tofile(file)
            with open(os.path.join(new_dataset_path,'train','images',patch_name.replace('.pkl','.bin')),'wb') as file:
                image.tofile(file)

            train_idx +=1
            
        elif (idx>= train_max_idx) and (idx< val_max_idx) :

            with open(os.path.join(new_dataset_path,'val','masks',mask_name.replace('_mask_weakly.pkl','_mask.bin')),'wb') as file:
                mask.tofile(file)
            with open(os.path.join(new_dataset_path,'val','images',patch_name.replace('.pkl','.bin')),'wb') as file:
                image.tofile(file)
            
            val_idx +=1
                

        elif (idx>= val_max_idx) and (idx<= test_max_idx):

            with open(os.path.join(new_dataset_path,'test','masks',mask_name.replace('_mask_weakly.pkl','_mask.bin')),'wb') as file:
                mask.tofile(file)
            with open(os.path.join(new_dataset_path,'test','images',patch_name.replace('.pkl','.bin')),'wb') as file:
                image.tofile(file)
            
            test_idx +=1


    print(f'    Events generated: {train_idx} TRAINING, {val_idx} VAL, and {test_idx} TEST ')


def split_notevents(new_dataset_path: str,
                    dataset_path: str = DATASET_PATH,
                    input_band_combination: list = ["B12","B11","B8A"],                    
                    train_split_ratio: float = 0.8,
                    val_split_ratio: float = 0.1,
                    test_split_ratio: float = 0.1,
                    seed: int = 42
                    ):
    ### Dataset path corresponds to the original dataset of the images

    random.seed(seed)

    if train_split_ratio+val_split_ratio+test_split_ratio != 1:
        if not test_split_ratio:
            test_split_ratio = 1 - (train_split_ratio + val_split_ratio)
        
        if train_split_ratio and not val_split_ratio and not test_split_ratio:
            print('Validation and test split ratios were not defined. Assuming equal split')
            val_split_ratio = (1-(train_split_ratio))/2
            test_split_ratio = val_split_ratio


    masks_path = os.path.join(dataset_path,'masks','weakly_segmentation')

    notevents_path = os.path.join(dataset_path,'images','notevent','NIR_SWIR')
    if not new_dataset_path:
        new_dataset_path = os.path.join(os.path.dirname(dataset_path),'train_random_split_dataset')
        create_dataset_folders(new_dataset_path)

    test_max_cont_end_loop = len(os.listdir(masks_path)) # Last index of where the data will be gathered
    train_max_cont_end_loop = np.round(test_max_cont_end_loop*train_split_ratio).astype(int)

    val_max_cont_end_loop = train_max_cont_end_loop + np.round(test_max_cont_end_loop*val_split_ratio).astype(int)

    mask = np.zeros((256,256,1),dtype=np.float32)

    with open(os.path.join(dataset_path,'granules_completed.txt'),'r') as file:
        # n_granules = len(file.readlines())
        granules = [line.strip() for line in file]

    n_scenes = len(np.unique([re.match(r'(.+)_G',granule).group(1) for granule in granules]))

    n_images_per_granule = np.round(test_max_cont_end_loop/n_scenes)

    notevents_paths_shuffle = os.listdir(notevents_path)
    random.shuffle(notevents_paths_shuffle) 

    cont_train = 0
    cont_val = 0
    cont_test = 0
    extra_images = []

    cont_end_loop = 1
    scene_names_list = []
    for image_name in notevents_paths_shuffle:
            
        scene_name = re.match(r'(.+)_G',image_name).group(1)

        if scene_names_list.count(scene_name)<n_images_per_granule: # This tries to get images from diverse scenes, to avoid getting only images from one scene
                        
            scene_names_list.append(scene_name)

            final_image = []
            for band in input_band_combination:
                for band_name,band_values in zip(band_combinations_dict.keys(),band_combinations_dict.values()):
                    if band in band_values:

                        patch_name = image_name.replace('NIR_SWIR',band_name)
                        image_path = os.path.join(dataset_path,'images','notevent',band_name,patch_name)
                        with open(image_path,'rb') as image_file:
                            image = pickle.load(image_file)

                        final_image.append(image[:,:,band_values.index(band)])
                        break

            image = np.transpose(np.array(final_image),(1,2,0))

            # with open(os.path.join(notevents_path,image_name),'rb') as image_file:
            #     image = pickle.load(image_file)

            if cont_end_loop<= train_max_cont_end_loop:

                with open(os.path.join(new_dataset_path,'train','masks',image_name.replace('_NIR_SWIR.pkl','_mask.bin')),'wb') as file:
                    mask.tofile(file)
                with open(os.path.join(new_dataset_path,'train','images',image_name.replace('_NIR_SWIR.pkl','.bin')),'wb') as file:
                    image.tofile(file)

                cont_end_loop +=1
                cont_train += 1

            elif (cont_end_loop> train_max_cont_end_loop) and (cont_end_loop<= val_max_cont_end_loop) :

                with open(os.path.join(new_dataset_path,'val','masks',image_name.replace('_NIR_SWIR.pkl','_mask.bin')),'wb') as file:
                    mask.tofile(file)
                with open(os.path.join(new_dataset_path,'val','images',image_name.replace('_NIR_SWIR.pkl','.bin')),'wb') as file:
                    image.tofile(file)
                
                cont_end_loop +=1
                cont_val += 1
            
            
            elif (cont_end_loop> val_max_cont_end_loop) and (cont_end_loop<= test_max_cont_end_loop):

                with open(os.path.join(new_dataset_path,'test','masks',image_name.replace('_NIR_SWIR.pkl','_mask.bin')),'wb') as file:
                    mask.tofile(file)
                with open(os.path.join(new_dataset_path,'test','images',image_name.replace('_NIR_SWIR.pkl','.bin')),'wb') as file:
                    image.tofile(file)

                cont_end_loop +=1
                cont_test += 1

            else:
                print('Reached maximum number of not events for the dataset')
                break
        else:
            extra_images.append(image_name)

    ### It is not able to get all the images for the testing from the different granules, and new images need to be included    
    missing_test_images = np.abs(test_max_cont_end_loop-val_max_cont_end_loop - cont_test)
    # print(f'Missing test images: {missing_test_images}')
    if missing_test_images>2:
        random.shuffle(extra_images)
        for i,image_name in enumerate(extra_images):
            if i == missing_test_images:
                break
            else:

                final_image = []
                for band in input_band_combination:
                    for band_name,band_values in zip(band_combinations_dict.keys(),band_combinations_dict.values()):
                        if band in band_values:

                            patch_name = image_name.replace('NIR_SWIR',band_name)
                            image_path = os.path.join(dataset_path,'images','notevent',band_name,patch_name)
                            with open(image_path,'rb') as image_file:
                                image = pickle.load(image_file)

                            final_image.append(image[:,:,band_values.index(band)])
                            break

                image = np.transpose(np.array(final_image),(1,2,0))

                with open(os.path.join(new_dataset_path,'test','masks',image_name.replace('_NIR_SWIR.pkl','_mask.bin')),'wb') as file:
                        mask.tofile(file)

                with open(os.path.join(new_dataset_path,'test','images',image_name.replace('.pkl','.bin')),'wb') as file:
                    image.tofile(file)
                cont_test += 1

    print(f'Not events generated: {cont_train} TRAINING, {cont_val} VAL, and {cont_test} TEST ')


def generate_random_split_dataset(new_dataset_path : str = None,
                                  dataset_path : str = DATASET_PATH,
                                  weakly : int = 0,
                                  band_combination: list = ["B12","B11","B8A"],                    
                                  train_split_ratio : float = 0.8,
                                  val_split_ratio : float = 0.1,
                                  test_split_ratio : float = 0.1,
                                  seed : int = 42
                                  ):
    
    if new_dataset_path:
        create_dataset_folders(new_dataset_path)

    split_events(dataset_path=dataset_path,
                 new_dataset_path = new_dataset_path,
                 weakly=weakly,
                 input_band_combination=band_combination,
                 val_split_ratio= val_split_ratio,
                 seed=seed)

    split_notevents(dataset_path = dataset_path,
                    new_dataset_path=new_dataset_path,
                    input_band_combination=band_combination,
                    train_split_ratio=train_split_ratio,
                    val_split_ratio=val_split_ratio,
                    test_split_ratio=test_split_ratio,
                    seed=seed)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training dataset generator')

    parser.add_argument('--new_dataset_path',   type=str,   help='Path where the new dataset wants to be generated.',   default=None)
    parser.add_argument('--dataset_path',       type=str,   help='Path of the full dataset with all the patches',       default=DATASET_PATH)
    parser.add_argument('--weakly',             type=int,   help='Use weakly supervision. 1: YES. 0: NO ',              default=1,choices=[0,1])
    parser.add_argument('--band_list',          type=str,   help='Specify the band combination for the dataset creation. Example: ["B12","B11","B8A"] ', nargs='+', default=["B12","B11","B8A"])
    parser.add_argument('--train_split_ratio',  type=float, help='Desired percentage for the training split.',          default=0.8)
    parser.add_argument('--val_split_ratio',    type=float, help='Desired percentage for the validation split.',        default=0.1)
    parser.add_argument('--test_split_ratio',   type=float, help='Desired percentage for the testing split.',           default=0.1)
    parser.add_argument('--seed',               type=int,   help='Seed for random number generators.',                  default=42)
    
    args = parser.parse_args()

    new_dataset_path = args.new_dataset_path
    dataset_path = args.dataset_path
    weakly = bool(args.weakly)
    train_split_ratio = args.train_split_ratio
    val_split_ratio = args.val_split_ratio
    test_split_ratio = args.test_split_ratio
    seed = args.seed

    if weakly:
        dataset_name = 'train_random_split_weakly_dataset'
    else:
        dataset_name = 'train_random_split_dataset'

    if new_dataset_path:
        create_dataset_folders(new_dataset_path)
    else:
        new_dataset_path = os.path.join(os.path.dirname(dataset_path),dataset_name)
        create_dataset_folders(new_dataset_path)

    if new_dataset_path:
        create_dataset_folders(new_dataset_path)

    split_events(dataset_path=dataset_path,
                 new_dataset_path = new_dataset_path,
                 weakly=weakly,
                 val_split_ratio= val_split_ratio,
                 seed=seed)

    split_notevents(dataset_path = dataset_path,
                    new_dataset_path=new_dataset_path,
                    train_split_ratio=train_split_ratio,
                    val_split_ratio=val_split_ratio,
                    test_split_ratio=test_split_ratio,
                    seed=seed)
