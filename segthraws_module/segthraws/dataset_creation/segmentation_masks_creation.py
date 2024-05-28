
"""
This script generates the segmentation masks for the weakly segmentation training. 
In this case, the events are saved as pixels with 255 values, whereas the potential_events are saved as pixels with 122 values

"""

import os
import pickle
import numpy as np
from .constants import DATASET_PATH

voting_4_masks = os.path.join(DATASET_PATH,'masks','event','comparison','voting_4')

segmentation_masks_path = os.path.join(os.path.dirname(os.path.dirname(voting_4_masks)),'segmentation_masks')
os.makedirs(segmentation_masks_path,exist_ok=True)

for mask_name in os.listdir(voting_4_masks):
    voting_4_mask_path = os.path.join(voting_4_masks,mask_name)
    voting_2_mask_path = os.path.join(os.path.dirname(voting_4_masks),'voting_2',mask_name.replace('_voting_4','_voting_2'))
    voting_3_mask_path = os.path.join(os.path.dirname(voting_4_masks),'voting_3',mask_name.replace('_voting_4','_voting_3'))
        
    with open(voting_4_mask_path,'rb') as voting_4_file:
        mask_voting_4 = pickle.load(voting_4_file)    

    with open(voting_3_mask_path,'rb') as voting_3_file:
        mask_voting_3 = pickle.load(voting_3_file)    
        
    with open(voting_2_mask_path,'rb') as voting_2_file:
        mask_voting_2 = pickle.load(voting_2_file)    
    
    new_mask = mask_voting_4.copy()

    condition_potential_event = np.logical_and(mask_voting_4 == 0,np.logical_or(mask_voting_2 == 255, mask_voting_3 == 255))

    new_mask[condition_potential_event] = 122

    new_mask_path = os.path.join(segmentation_masks_path,mask_name.replace('_voting_4','_weakly'))
    # print(new_mask_path)
    with open(new_mask_path,'wb') as new_mask_file:
        pickle.dump(new_mask,new_mask_file)

    