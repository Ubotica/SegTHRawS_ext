
"""
This script generates the statistics of the number of events in the segmentation masks. 

The columns of the csv output file represent: Patch name, number of event pixels, number of potential event pixels, number of not event pixels,
percentage of events pixels, percentage of potential events pixels, and percentage of not events pixels. 

The last three rows are the mean, standard deviation and the total sum of each column.

Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import pickle
from constants import DATASET_PATH
import numpy as np
import re
from csv import writer

segmentation_masks = os.path.join(DATASET_PATH,'masks','weakly_segmentation')

csv_headers = ['Name ','Event pixels ','Potential event pixels ','Not event pixels ','Proportion events ','Proportion potential events ','Proportion not events ']

csv_path = os.path.join(DATASET_PATH,'event_statistics.csv')

# Write the headers of the csv file
with open(csv_path, 'w') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(csv_headers)
    f_object.close()


statistic_list = []

for mask_name in os.listdir(segmentation_masks):
    segmentation_mask_path = os.path.join(segmentation_masks,mask_name)
    
    with open(segmentation_mask_path,'rb') as segmentation_file:
        mask_segmentation = pickle.load(segmentation_file)    
    
    n_event_pixels = np.sum(mask_segmentation[:,:,0]==1)
    n_potential_event_pixels = np.sum(mask_segmentation[:,:,0]==-1)
    n_not_event_pixels = np.sum(mask_segmentation[:,:,0]==0)

    proportion_event_pixels = np.round(n_event_pixels/ (n_event_pixels + n_potential_event_pixels + n_not_event_pixels)*100,3) 
    proportion_potential_event_pixels = np.round(n_potential_event_pixels/ (n_event_pixels + n_potential_event_pixels + n_not_event_pixels)*100,3)
    proportion_not_event_pixels = np.round(n_not_event_pixels/ (n_event_pixels + n_potential_event_pixels + n_not_event_pixels)*100,3)

    statistic_list.append([n_event_pixels,n_potential_event_pixels,n_not_event_pixels,proportion_event_pixels,proportion_potential_event_pixels,proportion_not_event_pixels])

    with open(csv_path, 'a') as f_object:
    
        writer_object = writer(f_object)
        writer_object.writerow([re.match(r'(.+)_mask',mask_name).group(1),n_event_pixels,n_potential_event_pixels,n_not_event_pixels,
                                proportion_event_pixels,proportion_potential_event_pixels,proportion_not_event_pixels])
    
        f_object.close()



with open(csv_path, 'a') as f_object:

    writer_object = writer(f_object)

    writer_object.writerow(['Mean values ',np.round(np.mean(statistic_list,axis=0),3)[0],
                                            np.round(np.mean(statistic_list,axis=0),3)[1],
                                            np.round(np.mean(statistic_list,axis=0),3)[2],
                                            np.round(np.mean(statistic_list,axis=0),3)[3],
                                            np.round(np.mean(statistic_list,axis=0),3)[4],
                                            np.round(np.mean(statistic_list,axis=0),3)[5]])

    writer_object.writerow(['Std values ', np.round(np.std(statistic_list,axis=0),3)[0],
                                            np.round(np.std(statistic_list,axis=0),3)[1],
                                            np.round(np.std(statistic_list,axis=0),3)[2],
                                            np.round(np.std(statistic_list,axis=0),3)[3],
                                            np.round(np.std(statistic_list,axis=0),3)[4],
                                            np.round(np.std(statistic_list,axis=0),3)[5]])

    writer_object.writerow(['Total sum ',   int(np.sum(statistic_list,axis=0)[0]),
                                            int(np.sum(statistic_list,axis=0)[1]),
                                            int(np.sum(statistic_list,axis=0)[2])])


    f_object.close()
    