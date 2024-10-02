"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from patchify import patchify



from ..utils import normalize_to_0_to_1
from .coregistration_superglue_multiband import SuperGlue_registration

from .constants import bands_list, thraws_data_path

def get_granule_image(desired_scene_name:str,
                      desired_granule_idx:int,
                      bands_list: list = bands_list,
                      data_path:str = thraws_data_path,
                      get_patches: bool = False,
                      visualize: bool = False):

    scene_groups_paths = [
            os.path.join(data_path,d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
        ]

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
                            raw_coreg_granule,_,coarse_status = SuperGlue_registration(bands_list=bands_list,event_path=scene_path,granule_idx=granule_idx)

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
