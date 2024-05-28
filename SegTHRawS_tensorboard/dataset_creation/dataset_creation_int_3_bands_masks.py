import os
import re
import sys
import matplotlib.pyplot as plt
import gc
import cv2
from multiprocessing import Process
import numpy as np


sys.path.insert(1, os.path.dirname(os.path.dirname(__file__))) 
# sys.path.insert(1, os.path.join(os.path.dirname(__file__),"..")) 

#### NEED TO MODIFY THIS TO INCLUDE PyRawS in the main directory of the SEGTHRAWS folder
sys.path.insert(1, '/home/cristopher/Documents/PyRawS/') # Include the PyRaws main folder

import pickle
from patchify import patchify

from pyraws.raw.raw_event import Raw_event

from s2pix_detector import s2pix_detector

from coregistration_superglue_multiband import SuperGlue_registration
from threshold_conditions import Castro_Traba_conditions,Massimetti_conditions,Murphy_conditions,Schroeder_conditions,Kumar_Roy_conditions

from utils import normalize_to_0_to_1
# from constants import *

from constants import SEGTHRAWS_DIRECTORY, MAIN_DIRECTORY, paths_dict, masks_events_dirs, masks_potential_events_dirs, bands_list, plot_names

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
warnings.filterwarnings('ignore',category=UserWarning)

# MAIN_DIRECTORY = os.path.dirname(__file__)


from matplotlib import font_manager
font_dirs = [os.path.join(SEGTHRAWS_DIRECTORY,'fonts','charter')]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

import matplotlib as mpl
mpl.rc('font',family='Charter')


# pyraws_path = os.path.dirname(os.path.dirname(MAIN_DIRECTORY))

# data_path = os.path.join(pyraws_path, "data", "raw")

data_path = '/home/cristopher/Documents/PyRawS/data/raw/'

break_condition = False


mask_generation_functions = (Castro_Traba_conditions,Massimetti_conditions,Murphy_conditions,Schroeder_conditions,Kumar_Roy_conditions)


def plot_mask(ax, mask, title):
    ax.imshow(mask)
    ax.set_title(title)



def scene_group_process(scenes_paths):
    for scene_path in scenes_paths:
        granules_paths = [
            os.path.join(scene_path,granule_name) for granule_name in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, granule_name))
        ]
        scene_name = os.path.basename(scene_path)
        

        if scene_name[-3:]!='_NE':
            print(f'{scene_name} started.')
            for granule_idx,_ in enumerate(granules_paths):

                raw_coreg_granule,coarse_status = SuperGlue_registration(bands_list=bands_list,event_path=scene_path,granule_idx=granule_idx)
                raw_coreg_granule = raw_coreg_granule.as_tensor().numpy()

                raw_coreg_granule_B01,_ = SuperGlue_registration(bands_list=[bands_list[0],"B01"],event_path=scene_path,granule_idx=granule_idx)

                B01_image = normalize_to_0_to_1(raw_coreg_granule_B01.as_tensor().numpy()[:,:,1])
                B01_image = cv2.resize(B01_image, (B01_image.shape[1],3*B01_image.shape[0]), interpolation=cv2.INTER_LINEAR)
                B01_patches = patchify(B01_image, (256,256), step=192)
                
                if coarse_status: # Check if the granule was coarse co-registered
                    with open(os.path.join(MAIN_DIRECTORY,'dataset','granules_coarse_coregistered.txt'),'a') as file:
                        file.write(f'{scene_name}_G_{granule_idx} \n')
                    coarse_status=False

                multispectral_patches = patchify(raw_coreg_granule,(256,256,raw_coreg_granule.shape[2]),step=192) 
                for i in range(multispectral_patches.shape[0]):
                    for j in range(multispectral_patches.shape[1]):
                        
                        patch_name = f'{scene_name}_G{granule_idx}_({j*192}, {i*192}, {256+j*192}, {256+i*192})'
                        NIR_SWIR_name = f'{patch_name}_NIR_SWIR'
                        RGB_name = f'{patch_name}_RGB'

                        NIR_SWIR_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B12')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B11')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B8A')])))

                        RGB_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B04')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B03')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B02')])))

                        Vegetation_Red_patch = normalize_to_0_to_1(np.dstack((multispectral_patches[i,j,0,:,:,bands_list.index('B07')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B06')],
                                                    multispectral_patches[i,j,0,:,:,bands_list.index('B05')])))

                        NIR_1_patch = normalize_to_0_to_1(multispectral_patches[i,j,0,:,:,1].reshape(256,256,1))
                        B01_patch = B01_patches[i,j,:,:].reshape(256,256,1)

                        if not (NIR_SWIR_patch.max() == 0 or RGB_patch.max() == 0 or Vegetation_Red_patch.max() == 0 or NIR_1_patch.max() == 0):
                            
                            image_multiband = np.dstack((NIR_SWIR_patch,RGB_patch,B01_patch))
                            image_masks = []

                            for k,mask_generator in enumerate(mask_generation_functions):
                                if k<=2:
                                    image_masks.append(mask_generator(NIR_SWIR_patch))
                                else:
                                    image_masks.append(mask_generator(image_multiband))
                            
                            combined_masks = (np.dstack((image_masks[0][:,:,0],
                                        image_masks[1][:,:,0],
                                        image_masks[2][:,:,0],
                                        image_masks[3][:,:,0],
                                        image_masks[4][:,:,0]))/255).astype(np.uint8)
                    
                            intersection = np.stack([np.all(combined_masks>0,axis =-1).astype(np.uint8)*255] * 3, axis=-1) # Convert to three band mask
                            voting_2 = np.stack([(((np.sum(combined_masks,axis=-1)))>=2).astype(np.uint8)*255] * 3, axis=-1) # Convert to three band mask
                            voting_3 = np.stack([(((np.sum(combined_masks,axis=-1)))>=3).astype(np.uint8)*255] * 3, axis=-1) # Convert to three band mask
                            voting_4 = np.stack([(((np.sum(combined_masks,axis=-1)))>=4).astype(np.uint8)*255] * 3, axis=-1) # Convert to three band mask

                            if np.any(voting_2): #Intersection and voting 3 not included, because voting_2 condition is enough to consider an event.
                                
                                # Creation of a comparison image for the performance detection of the different masks
                                plt.figure(num=1,figsize=(15, 6),clear=True)
                                plt.suptitle(f'{patch_name}', fontsize=14)
                                for k, mask in enumerate([NIR_SWIR_patch, *image_masks, voting_2, voting_3, voting_4, intersection],1):
                                    plot_mask(plt.subplot(2,5,k), mask, plot_names[k-1])
                                plt.tight_layout()

                                additional_event_condition = np.any(voting_4) #An event is identified if Voting 4 is true.

                                if not additional_event_condition:
                                    
                                    with open(os.path.join(paths_dict['potential_events_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                            pickle.dump(NIR_SWIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['potential_events_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                        pickle.dump(RGB_patch, f)
                                    
                                    with open(os.path.join(paths_dict['potential_events_images_Vegetation_path'],patch_name+'_VEG.pkl'),'wb') as f:
                                            pickle.dump(Vegetation_Red_patch, f)
                                        
                                    with open(os.path.join(paths_dict['potential_events_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                        pickle.dump(NIR_1_patch, f)

                                    for k,mask in enumerate(image_masks):
                                        if np.any(mask):
                                            with open(os.path.join(masks_potential_events_dirs[k],NIR_SWIR_name+'_mask.pkl'),'wb') as f:
                                                pickle.dump(mask, f)
                                    
                                    with open(os.path.join(paths_dict['masks_potential_event_voting_2_path'],patch_name+'_mask_voting_2.pkl'),'wb') as file:
                                        pickle.dump(voting_2,file)
                                    plt.savefig(os.path.join(paths_dict['masks_comparison_potential_events_plot_voting_2_path'], f'{patch_name}_comparison.png'))

                                    if np.any(voting_3):
                                        with open(os.path.join(paths_dict['masks_potential_event_voting_3_path'],patch_name+'_mask_voting_3.pkl'),'wb') as file:
                                            pickle.dump(voting_3,file)
                                        plt.savefig(os.path.join(paths_dict['masks_comparison_potential_events_plot_voting_3_path'], f'{patch_name}_comparison.png'))
                                    plt.close()

                                else: #Events condition (voting_4 exists)

                                    with open(os.path.join(paths_dict['events_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                            pickle.dump(NIR_SWIR_patch, f)
                                        
                                    with open(os.path.join(paths_dict['events_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                        pickle.dump(RGB_patch, f)

                                    with open(os.path.join(paths_dict['events_images_Vegetation_path'],patch_name+'_VEG.pkl'),'wb') as f:
                                        pickle.dump(Vegetation_Red_patch, f)
                                        
                                    with open(os.path.join(paths_dict['events_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                        pickle.dump(NIR_1_patch, f)

                                    for k,mask in enumerate(image_masks):
                                        if np.any(mask):
                                            with open(os.path.join(masks_events_dirs[k],NIR_SWIR_name+'_mask.pkl'),'wb') as f:
                                                pickle.dump(mask, f)

                                    with open(os.path.join(paths_dict['masks_event_voting_2_path'],patch_name+'_mask_voting_2.pkl'),'wb') as file:
                                        pickle.dump(voting_2,file)

                                    # plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_2_path'], f'{patch_name}_comparison.png')) #Unnecessary
                                      
                                    with open(os.path.join(paths_dict['masks_event_voting_3_path'],patch_name+'_mask_voting_3.pkl'),'wb') as file:
                                        pickle.dump(voting_3,file)

                                    # plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_3_path'], f'{patch_name}_comparison.png')) #Unnecessary

                                    with open(os.path.join(paths_dict['masks_event_voting_4_path'],patch_name+'_mask_voting_4.pkl'),'wb') as file:
                                        pickle.dump(voting_4,file)
                                    plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_voting_4_path'], f'{patch_name}_comparison.png'))

                                    if np.any(intersection): #Intersection only happens if voting_4 exists
                                        with open(os.path.join(paths_dict['masks_event_intersection_path'],patch_name+'_mask_intersection.pkl'),'wb') as file:
                                            pickle.dump(intersection,file)
                                        plt.savefig(os.path.join(paths_dict['masks_comparison_events_plot_intersection_path'], f'{patch_name}_comparison.png'))
                                    
                                    plt.close()
                            else: #Not event

                                with open(os.path.join(paths_dict['notevents_images_NIR_SWIR_path'],NIR_SWIR_name+'.pkl'),'wb') as f:
                                    pickle.dump(NIR_SWIR_patch, f)
                                
                                with open(os.path.join(paths_dict['notevents_images_RGB_path'],RGB_name+'.pkl'),'wb') as f:
                                    pickle.dump(RGB_patch, f)
                                    
                                with open(os.path.join(paths_dict['notevents_images_Vegetation_path'],patch_name+'_VEG.pkl'),'wb') as f:
                                    pickle.dump(Vegetation_Red_patch, f)
                                        
                                with open(os.path.join(paths_dict['notevents_images_NIR1_path'],patch_name+'_NIR1.pkl'),'wb') as f:
                                    pickle.dump(NIR_1_patch, f)
                            
                            gc.collect()

                with open(os.path.join(MAIN_DIRECTORY, 'dataset','granules_completed.txt'),'a') as file:
                    file.write(f'{scene_name}_G_{granule_idx}\n')
            print(f'{scene_name} completed.')
            



scene_groups_paths = [
            os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
        ]



process_list = []

for idx_process,scene_group_path in enumerate(scene_groups_paths):

    if idx_process <=1:

        scenes_paths = [
                os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
            ]
        # print('a',idx_process,scene_group_path)     
        process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))


# Process starting loop

for process in process_list:
    process.start()

for process in process_list:
    process.join()


for process_groups in range(1,6):
    process_list = []


    # Process creating loop
    if process_groups == 5: #This accounts for the last 2 process
        for idx_process,scene_group_path in enumerate(scene_groups_paths):
        
            if idx_process<2+process_groups*3 and idx_process>=process_groups*3:

                scenes_paths = [
                        os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
                    ]
                # print('a',idx_process,scene_group_path)     
                process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))
    else:
        for idx_process,scene_group_path in enumerate(scene_groups_paths):
            
            if idx_process<=2+process_groups*3 and idx_process>=process_groups*3:
                scenes_paths = [
                        os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
                    ]
                # print('b',idx_process,scene_group_path)     
                process_list.append(Process(target=scene_group_process,args=(scenes_paths,)))

    # Process starting loop

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

