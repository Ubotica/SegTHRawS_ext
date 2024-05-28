
import os
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify

import pickle

def normalize_to_0_to_1(img):
    """Normalizes the passed image to 0 to 1

    Args:
        img (np.array): image to normalize

    Returns:
        np.array: normalized image
    """
    # img = img + np.minimum(0, np.min(img))  # move min to 0
    # img = img / np.max(img)  # scale to 0 to 1
    img = img / 4095  # scale to 0 to 1
    return img


def normalize(band):
    # Function that converts 16bits images into 8bits images
    band_max, band_min = band.max(), band.min()
    return (((band - band_min) / ((band_max - band_min))) * 255).astype(np.uint8)


from coregistration_superglue_multiband import SuperGlue_registration

bands_list = [ "B02", "B08", "B03", "B04", "B05","B11", "B06", "B07","B8A","B12"]


data_path = '/home/cristopher/Documents/PyRawS/data/raw'


def get_granule_image(desired_scene_name:str,
                      desired_granule_idx:int,
                      bands_list: list = bands_list,
                      data_path:str = data_path,
                      get_patches: bool = False,
                      visualize: bool = False):

    scene_groups_paths = [
            os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
        ]

    # scene_name_condition = False
    # Process creating loop
    for i,scene_group_path in enumerate(scene_groups_paths):
        
            scenes_paths = [
                    os.path.join(scene_group_path,scene_name) for scene_name in os.listdir(scene_group_path) if os.path.isdir(os.path.join(scene_group_path, scene_name))
                ]
            
            for scene_path in scenes_paths:
                granules_paths = [
                    os.path.join(scene_path,granule_name) for granule_name in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, granule_name))
                ]
                scene_name = os.path.basename(scene_path)

                if scene_name == desired_scene_name:
                    # print(scene_name)
                    for granule_idx in range(len(granules_paths)):

                        if granule_idx == desired_granule_idx:
                            raw_coreg_granule,coarse_status = SuperGlue_registration(bands_list=bands_list,event_path=scene_path,granule_idx=granule_idx)

                            raw_coreg_granule = raw_coreg_granule.as_tensor().numpy()

                            NIR_SWIR_image = normalize_to_0_to_1(np.dstack((raw_coreg_granule[:,:,9],raw_coreg_granule[:,:,5],raw_coreg_granule[:,:,8])))

                            NIR_SWIR_patches = patchify(NIR_SWIR_image, (256,256,3), step=192)
                            # print(NIR_SWIR_patches.shape)

                            if visualize:
                                plt.figure()
                                plt.imshow(NIR_SWIR_image)
                                plt.title(f'{scene_name}_G{granule_idx}')
                                plt.show()

                            if get_patches:


                                for i in range(NIR_SWIR_patches.shape[0]):
                                    for j in range(NIR_SWIR_patches.shape[1]):


                                        patch_name = f'{scene_name}_G{granule_idx}_({j*192}, {i*192}, {256+j*192}, {256+i*192})'
                                        print(patch_name)
                                        NIR_SWIR_name = f'{patch_name}_NIR_SWIR'

                                        NIR_SWIR_patch = NIR_SWIR_patches[i,j,:,:,:].reshape(256,256,3)
                                        
                                        print(f'{patch_name}, MAX: {NIR_SWIR_patch.max()} MIN: {NIR_SWIR_patch.min()}')
                                        if visualize:
                                            plt.figure()
                                            plt.imshow(NIR_SWIR_patch)
                                            plt.title(NIR_SWIR_name)
                                            plt.show()
                                
                                return NIR_SWIR_patches
                            else:
                                return NIR_SWIR_image
                            # with open(os.path.join(os.getcwd(),NIR_SWIR_name+'.pkl'),'wb') as f:
                            #     pickle.dump(NIR_SWIR_patch, f)

                            # with open(os.path.join(os.getcwd(),NIR_SWIR_name+'.pkl'),'rb') as f:
                            #     saved_patch = pickle.load(f)
                            #     print(f'Saved patch: MAX {saved_patch.max()} MIN {saved_patch.min()}')


            #                 scene_name_condition = True
            #                 break
            #         if scene_name_condition:
            #             break
            # if scene_name_condition:
            #     break