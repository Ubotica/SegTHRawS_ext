"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import gc
import cv2
import sys
import pickle
import argparse
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from patchify import patchify
from matplotlib import font_manager
from multiprocessing import Process


from .coregistration_superglue_multiband import SuperGlue_registration


from ..utils import normalize_to_0_to_1

from .constants import SEGTHRAWS_DIRECTORY, DATASET_PATH,thraws_data_path
from .constants import paths_dict, masks_events_dirs, masks_potential_events_dirs
from .constants import bands_list, plot_names, mask_generation_functions 
from .constants import PATCH_SIZE, PATCH_STEP

warnings.filterwarnings('ignore',category=RuntimeWarning)
warnings.filterwarnings('ignore',category=UserWarning)


font_dirs = [os.path.join(SEGTHRAWS_DIRECTORY,'fonts','charter')]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

mpl.rc('font',family='Charter')

def plot_mask(ax, mask, title):
    ax.imshow(mask)
    ax.set_title(title)

def scene_group_process(scenes_paths: str,
                        dataset_path: str = DATASET_PATH,
                        downsampling: bool = True,
                        ):
    for scene_path in scenes_paths:
        granules_paths = [
            os.path.join(scene_path,granule_name) for granule_name in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, granule_name))
        ]
        scene_name = os.path.basename(scene_path)
        

        if scene_name[-3:]!='_NE':
            print(f'{scene_name} started.')
            for granule_idx,_ in enumerate(granules_paths):

                raw_coreg_granule,raw_coordinates,coarse_status = SuperGlue_registration(bands_list=bands_list,event_path=scene_path,granule_idx=granule_idx)
                raw_coreg_granule = raw_coreg_granule.as_tensor(downsampling=downsampling).numpy() #Produces negative values in the images

                raw_coreg_granule_B01,_,_ = SuperGlue_registration(bands_list=[bands_list[0],"B01"],event_path=scene_path,granule_idx=granule_idx)
                empty_pixels_pattern = np.all(raw_coreg_granule != 0,axis=2)

                B01_image = normalize_to_0_to_1(raw_coreg_granule_B01.as_tensor(downsampling=downsampling).numpy()[:,:,1])
                B01_image = cv2.resize(B01_image, (B01_image.shape[1],3*B01_image.shape[0]), interpolation=cv2.INTER_LINEAR)
                B01_patches = patchify(B01_image, (PATCH_SIZE,PATCH_SIZE), step=PATCH_STEP)
                
                if coarse_status: # Check if the granule was coarse co-registered
                    with open(os.path.join(dataset_path,'granules_coarse_coregistered.txt'),'a') as file:
                        file.write(f'{scene_name}_G_{granule_idx} \n')
                    coarse_status=False

                multispectral_patches = patchify(raw_coreg_granule,(PATCH_SIZE,PATCH_SIZE,raw_coreg_granule.shape[2]),step=PATCH_STEP) 
                empty_pixels_pattern_patches = patchify(empty_pixels_pattern,(PATCH_SIZE,PATCH_SIZE),step=PATCH_STEP) 

                # print(multispectral_patches.max(),multispectral_patches.min())
                for i in range(multispectral_patches.shape[0]):
                    for j in range(multispectral_patches.shape[1]):
                        
                        patch_name = f'{scene_name}_G{granule_idx}_({j*PATCH_STEP}, {i*PATCH_STEP}, {PATCH_SIZE+j*PATCH_STEP}, {PATCH_SIZE+i*PATCH_STEP})'
                        NIR_SWIR_name = f'{patch_name}_NIR_SWIR'
                        RGB_name = f'{patch_name}_RGB'

                        NIR_SWIR_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B12')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B11')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B8A')])))

                        RGB_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B04')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B03')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B02')])))

                        VNIR_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B07')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B06')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B05')])))

                        NIR_1_patch = normalize_to_0_to_1(multispectral_patches[i,j,0,:,:,bands_list.index('B08')].reshape(PATCH_SIZE,PATCH_SIZE,1))

                        B01_patch = B01_patches[i,j,:,:].reshape(PATCH_SIZE,PATCH_SIZE,1)
                        # B01_patch = normalize_to_0_to_1(multispectral_patches[i,j,0,:,:,bands_list.index('B01')].reshape(PATCH_SIZE,PATCH_SIZE,1))

                        empty_pixels_pattern_patch = empty_pixels_pattern_patches[i,j,:,:].reshape(PATCH_SIZE,PATCH_SIZE)

                        # print(NIR_SWIR_patch.max(),NIR_SWIR_patch.min())

                        if not (NIR_SWIR_patch.max() == 0 or RGB_patch.max() == 0 or VNIR_patch.max() == 0 or NIR_1_patch.max() == 0):
                            
                            image_multiband = np.dstack((NIR_SWIR_patch,RGB_patch,B01_patch))
                            image_masks = []



                            for k,mask_generator in enumerate(mask_generation_functions):
                                if k<=2:
                                    image_masks.append(mask_generator(NIR_SWIR_patch,empty_pixels_pattern_patch))
                                else:
                                    image_masks.append(mask_generator(image_multiband,empty_pixels_pattern_patch))
                            
                            combined_masks = (np.dstack((image_masks[0],
                                                         image_masks[1],
                                                         image_masks[2],
                                                         image_masks[3],
                                                         image_masks[4]))).astype(np.float32)
                    
                            voting_2 = (((np.sum(combined_masks,axis=-1)))>=2).astype(np.float32) # Single band mask
                            voting_3 = (((np.sum(combined_masks,axis=-1)))>=3).astype(np.float32) # Single band mask
                            voting_4 = (((np.sum(combined_masks,axis=-1)))>=4).astype(np.float32) # Single band mask
                            intersection = np.all(combined_masks>0,axis =-1).astype(np.float32) # Single band mask

                            if np.any(voting_2): # Voting_2 condition is enough to consider a potential event.
                                
                                # Creation of a comparison image for the performance detection of the different masks
                                plt.figure(num=1,figsize=(15, 6),clear=True)
                                plt.suptitle(f'{patch_name}', fontsize=14)
                                for k, mask in enumerate([NIR_SWIR_patch, *image_masks, voting_2, voting_3, voting_4, intersection],1):
                                    if mask.ndim == 2:
                                        mask = np.stack([mask] * 3, axis=-1)
                                    plot_mask(plt.subplot(2,5,k), mask, plot_names[k-1])
                                plt.tight_layout()

                                additional_event_condition = np.any(voting_4) #An event is identified if Voting 4 is true.

                                if not additional_event_condition:
                                    
                                    with open(os.path.join(paths_dict['potential_events_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                            pickle.dump(NIR_SWIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['potential_events_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                        pickle.dump(RGB_patch, f)
                                    
                                    with open(os.path.join(paths_dict['potential_events_images_VNIR_path'],patch_name+'_VNIR.pkl'),'wb') as f:
                                            pickle.dump(VNIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['potential_events_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                        pickle.dump(NIR_1_patch, f)

                                    for k,mask in enumerate(image_masks):
                                        if np.any(mask):
                                            mask = mask[:,:,np.newaxis] # Save the masks with shape (height,width,channel), where channels = 1
                                            with open(os.path.join(masks_potential_events_dirs[k],NIR_SWIR_name+'_mask.pkl'),'wb') as f:
                                                pickle.dump(mask, f)
                                    
                                    with open(os.path.join(paths_dict['masks_potential_event_voting_2_path'],patch_name+'_mask_voting_2.pkl'),'wb') as file:
                                        pickle.dump(voting_2[:,:,np.newaxis],file)
                                    plt.savefig(os.path.join(paths_dict['masks_comparison_potential_events_plot_voting_2_path'], f'{patch_name}_comparison.png'))

                                    if np.any(voting_3):
                                        with open(os.path.join(paths_dict['masks_potential_event_voting_3_path'],patch_name+'_mask_voting_3.pkl'),'wb') as file:
                                            pickle.dump(voting_3[:,:,np.newaxis],file)
                                        plt.savefig(os.path.join(paths_dict['masks_comparison_potential_events_plot_voting_3_path'], f'{patch_name}_comparison.png'))
                                    plt.close()

                                else: #Events condition (voting_4 exists)

                                    with open(os.path.join(paths_dict['events_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                            pickle.dump(NIR_SWIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['events_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                        pickle.dump(RGB_patch, f)

                                    with open(os.path.join(paths_dict['events_images_VNIR_path'],patch_name+'_VNIR.pkl'),'wb') as f:
                                        pickle.dump(VNIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['events_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                        pickle.dump(NIR_1_patch, f)

                                    for k,mask in enumerate(image_masks):
                                        if np.any(mask):
                                            mask = mask[:,:,np.newaxis] # Save the masks with shape (height,width,channel), where channels = 1
                                            with open(os.path.join(masks_events_dirs[k],NIR_SWIR_name+'_mask.pkl'),'wb') as f:
                                                pickle.dump(mask, f)

                                    with open(os.path.join(paths_dict['masks_event_voting_2_path'],patch_name+'_mask_voting_2.pkl'),'wb') as file:
                                        pickle.dump(voting_2[:,:,np.newaxis],file)

                                    # plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_2_path'], f'{patch_name}_comparison.png')) #Unnecessary
                                      
                                    with open(os.path.join(paths_dict['masks_event_voting_3_path'],patch_name+'_mask_voting_3.pkl'),'wb') as file:
                                        pickle.dump(voting_3[:,:,np.newaxis],file)

                                    # plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_3_path'], f'{patch_name}_comparison.png')) #Unnecessary

                                    with open(os.path.join(paths_dict['masks_event_voting_4_path'],patch_name+'_mask_voting_4.pkl'),'wb') as file:
                                        pickle.dump(voting_4[:,:,np.newaxis],file)
                                    plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_4_path'], f'{patch_name}_comparison.png'))

                                    if np.any(intersection): #Intersection only happens if voting_4 exists
                                        with open(os.path.join(paths_dict['masks_event_intersection_path'],patch_name+'_mask_intersection.pkl'),'wb') as file:
                                            pickle.dump(intersection[:,:,np.newaxis],file)
                                        plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_intersection_path'], f'{patch_name}_comparison.png'))
                                    
                                    ##### Weakly masks segmentation generation
                                    weakly_mask = voting_4.copy()
                                    condition_potential_event = np.logical_and(voting_4 == 0,np.logical_or(voting_2 == 1, voting_3 == 1))
                                    weakly_mask[condition_potential_event] = -1
                                    with open(os.path.join(paths_dict['masks_weakly_segmentation_path'],patch_name+'_mask_weakly.pkl'),'wb') as weakly_mask_file:
                                        pickle.dump(weakly_mask[:,:,np.newaxis],weakly_mask_file)

                                    plt.close()
                            else: #Not event

                                with open(os.path.join(paths_dict['notevents_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                    pickle.dump(NIR_SWIR_patch, f)
                                
                                with open(os.path.join(paths_dict['notevents_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                    pickle.dump(RGB_patch, f)
                                    
                                with open(os.path.join(paths_dict['notevents_images_VNIR_path'],patch_name+'_VNIR.pkl'),'wb') as f:
                                    pickle.dump(VNIR_patch, f)
                                        
                                with open(os.path.join(paths_dict['notevents_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                    pickle.dump(NIR_1_patch, f)
                            
                            gc.collect()

                with open(os.path.join(dataset_path,'granules_completed.txt'),'a') as file:
                    file.write(f'{scene_name}_G_{granule_idx}\n')
            print(f'{scene_name} completed.')
            

def dataset_creation_multiprocess(data_path: str):

    scene_groups_paths = [
                os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
            ]

    scene_groups_paths = sorted(scene_groups_paths, key=lambda x: int(x.rsplit('/', 1)[-1]))

    n_process = 5
    n_scene_groups = len(scene_groups_paths)
    # For the rest of the scenes, three simultaneous process can be performed
    for process_groups in range(0,np.round(n_scene_groups/n_process).astype(np.int8)+1):

        process_list = []

        # Process creating loop
        if process_groups == n_process: #This accounts for the last 2 process
            for idx_process,scene_group_path in enumerate(scene_groups_paths):
            
                if idx_process<(n_process-1)+process_groups*n_process and idx_process>=process_groups*n_process:

                    scenes_paths = [
                            os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
                        ]
                    process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))
        else:
            for idx_process,scene_group_path in enumerate(scene_groups_paths):
                
                if idx_process<=(n_process-1)+process_groups*n_process and idx_process>=process_groups*n_process:
                    scenes_paths = [
                            os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
                        ]

                    process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))

        # Process starting loop
        for process in process_list:
            process.start()

        for process in process_list:
            process.join()


    # # The first folders can only be run as a group of 2 for the current CPU
    # process_list = []

    # for idx_process,scene_group_path in enumerate(scene_groups_paths):

    #     if idx_process >=12:

    #         scenes_paths = [
    #                 os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
    #             ]
    #         # print('a',idx_process,scene_group_path)     
    #         process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))

    # for process in process_list:
    #     process.start()

    # for process in process_list:
    #     process.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',  type=str, help='Path to the raw THRawS data directory',default=thraws_data_path)

    # parse the arguments
    args = parser.parse_args()

    data_path = args.data_path

    dataset_creation_multiprocess(data_path=data_path)
