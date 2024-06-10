"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import torch
import numpy as np

from s2pix_detector import s2pix_detector


#####3 CHANGES: ADD THE B01 band to Schroeder and Kumar-Roy

def Castro_Traba_conditions(input_image,neighborhood_size = 30):

    input_image = input_image[:,:,::-1]

    mask0 = ~np.any(input_image == 0,axis=2)
    new_image = input_image*mask0[:,:,np.newaxis]

    #Murphy conditions (Massimetti uses mask2 >= 1.2)
    mask1 =  new_image[:,:,2]/new_image[:,:,1] >= 1.4
    mask2 =  new_image[:,:,2]/new_image[:,:,0] >= 1.4
    mask3 =  new_image[:,:,2] >= 0.15#*255



    # mask4 =  new_image[:,:,1]/new_image[:,:,0] >=2
    mask4 =  np.logical_and(new_image[:,:,2] >100/255,new_image[:,:,0] < 15/255)

    # Condition for surrounding clouds with high values of SWIR bands
    mask5 = np.logical_and(np.logical_or(new_image[:,:,2]>=150/255,new_image[:,:,1]>=150/255),new_image[:,:,0]<60/255)
    # mask5 = np.logical_or(new_image[:,:,2]>=150,new_image[:,:,1]>=150)

    ## Hottest pixel conditions (almost white)
    # mask6 = np.logical_and(np.logical_and(new_image[:,:,2]>=180,new_image[:,:,1]>=180),new_image[:,:,0]>125)
    mask6 = np.logical_and(new_image[:,:,2]>=180/255,new_image[:,:,1]>=180/255)
    
    #These 3 conditions can be in 1 line
    mask_combined = np.logical_and(mask1,np.logical_and(mask2,np.logical_and(mask3,mask4)))
    
    mask_combined = np.logical_or(mask_combined,mask5)

    mask_combined = np.logical_or(mask_combined,mask6)

    mask_combined_temp = mask_combined.copy()


    for i in range(mask_combined.shape[0]):
            for j in range(mask_combined.shape[1]):
                if mask_combined[i, j]:

                    neighbor_window = mask_combined[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                    ]

                    condition_1 = (
                        new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            2
                        ]
                        / new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            1
                        ]
                        >= 4
                    )

                    condition_2 = (
                        new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            0
                        ]
                        < 0.1
                    )
                    
                    mask_combined_temp[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                    ] = np.logical_or(
                        neighbor_window,  np.logical_and(condition_1, condition_2)
                        # neighbor_window,  condition_1
                    )

    mask_combined = mask_combined_temp

    # Condition for the inner pixels that were not detected before
    for i in range(mask_combined.shape[0]):
        for j in range(mask_combined.shape[1]):
            if np.sum(mask_combined[
                max(0, i - 1) : i + 2,
                max(0, j - 1) : j + 2,
            ]) >= 7:
                mask_combined[i,j]=True

    mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255

    return mask_broadcasted

def Murphy_conditions(input_image,neighborhood_size = 8):
    
    input_image = input_image[:,:,::-1]

    mask0 = ~np.any(input_image == 0,axis=2)
    new_image = input_image*mask0[:,:,np.newaxis]
    ### Murphy conditions initial fire pixels

    mask1 = new_image[:, :, 2] / new_image[:, :, 1] >= 1.4
    mask2 = new_image[:, :, 2] / new_image[:, :, 0] >= 1.4
    mask3 = new_image[:, :, 2] >= 0.15 

    mask_combined = np.logical_and(mask1, np.logical_and(mask2, mask3))


    ### Murphy conditions potential fire pixels

    mask_combined_temp = mask_combined.copy()

    for i in range(mask_combined.shape[0]):
        for j in range(mask_combined.shape[1]):
            if mask_combined[i, j]:


                neighbor_window = mask_combined[
                    max(0, i - neighborhood_size // 2) : i
                    + neighborhood_size // 2
                    + 1,
                    max(0, j - neighborhood_size // 2) : j
                    + neighborhood_size // 2
                    + 1,
                ]

                condition_1 = (
                    new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        1
                    ]
                    / new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        0
                    ]
                    >= 2
                )

                condition_2 = (
                    new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        1
                    ]
                    > 0.5
                )
                
                mask_combined_temp[
                    max(0, i - neighborhood_size // 2) : i
                    + neighborhood_size // 2
                    + 1,
                    max(0, j - neighborhood_size // 2) : j
                    + neighborhood_size // 2
                    + 1,
                ] = np.logical_or(
                    neighbor_window,  np.logical_and(condition_1, condition_2)
                )

    mask_combined = mask_combined_temp

    mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255

    return mask_broadcasted

def Massimetti_conditions(input_image):
    input_image_copy = input_image[:,:,::-1].copy() #Needed because s2pix detector expects the 8A band first
    _, raw_filtered_alert_matrix, _ = s2pix_detector(torch.Tensor(input_image_copy))

    mask_broadcasted = np.stack([raw_filtered_alert_matrix.bool()] * 3, axis=-1).astype(np.uint8)*255

    return mask_broadcasted

def Kumar_Roy_conditions(input_image):

    if input_image.shape[2] == 7:
        input_image = input_image[:,:,:-1] #This eliminates the B01 band
    neighborhood_sizes = [x for x in range(5,61+1,2)]

    ### Unambiguous active fires conditions

    mask1 = input_image[:, :, 3] <= 0.53 * input_image[:, :, 0] - 0.214


    mask1_temp = mask1.copy()

    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j]:

                neighbor_window = mask1[
                    max(0, i - 1) : i + 1 + 1,
                    max(0, j - 1) : j + 1 + 1,
                ]
                mask1_neighborhood = (
                    input_image[
                        max(0, i - 1) : i + 1 + 1,
                        max(0, j - 1) : j + 1 + 1,
                        3,
                    ]
                    <= 0.35
                    * input_image[
                        max(0, i - 1) : i + 1 + 1,
                        max(0, j - 1) : j + 1 + 1,
                        1,
                    ]
                    - 0.044
                )

                #Iterative process to all the detected pixels. The iteration does not account for pixels in smallers i or j, as it would be too computationally expensive 
                mask1_temp[
                    max(0, i - 1) : i + 1 + 1,
                    max(0, j - 1) : j + 1 + 1,
                ] = np.logical_or(neighbor_window, mask1_neighborhood)

    mask1 = mask1_temp

    ### Potential fire conditions

    mask2 = np.logical_or(
        input_image[:, :, 3] <= 0.53 * input_image[:, :, 0] - 0.125,
        input_image[:, :, 1] <= 1.08 * input_image[:, :, 0] - 0.048,
    )

    mask_combined_temp = mask2.copy()

    ### Verification of potential fires detected

    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i, j]:
                for neighborhood_size in neighborhood_sizes:

                    neighbor_indexes = (
                        slice(
                            max(0, i - neighborhood_size // 2), i + neighborhood_size // 2 + 1
                        ),
                        slice(
                            max(0, j - neighborhood_size // 2), j + neighborhood_size // 2 + 1
                        ),
                    )

                    neighbor_window = input_image[neighbor_indexes]

                    negated_mask = ~np.logical_and(
                        mask2[neighbor_indexes], mask1[neighbor_indexes]
                    )
                    
                    # This represents the image without the unambiguous active fire and potential fires pixels. Next step discard, water pixels
                    mask_window = neighbor_window * negated_mask[:, :, np.newaxis]

                    # Water pixel conditions
                    condition_water_pixels = np.logical_and(
                        neighbor_window[:, :, 5] > neighbor_window[:, :, 4],
                        np.logical_and(
                            neighbor_window[:, :, 4] > neighbor_window[:, :, 3],
                            neighbor_window[:, :, 4] > neighbor_window[:, :, 2],
                        ),
                    )

                    condition_water_negated = ~condition_water_pixels

                    if np.sum(condition_water_pixels)/condition_water_pixels.size < 0.25:
                        mask_combined_temp[i,j] = False # If never finds 25% of pixels, it is not considered a fire
                    else:
                        ### This represents the elements that are not fires or water pixels
                        mask_window = mask_window * condition_water_negated[:, :, np.newaxis]

                        condition_1 = input_image[i, j, 0] / input_image[i, j, 2] > np.mean(
                            mask_window[:,:,0]/mask_window[:,:,2]
                            + max(0.8, 3 * np.std(mask_window[:,:,0]/mask_window[:,:,2]))
                        )

                        condition_2 = input_image[i, j, 0] / input_image[i, j, 2] > np.mean(
                            mask_window[:,:,0]
                            + max(
                                0.08,
                                3
                                * np.std(mask_window[:,:,0]),
                            )
                        )
                        real_potential_fire_condition = np.logical_and(condition_1,condition_2)

                        if real_potential_fire_condition:
                            mask_combined_temp[i,j] = True
                        else:
                            mask_combined_temp[i,j] = False
                        break

    mask2 = mask_combined_temp

    mask_combined = np.logical_or(mask1, mask2)

    mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255


    return mask_broadcasted

def Schroeder_conditions(input_image,neighborhood_size = 61):

    ### The expected bands are: B12, B11, B8A, B04, B03, B02, and B01

    #Unambiguous active fire pixels

    mask1 = input_image[:, :, 0] / input_image[:, :, 2] > 2.5
    mask2 = input_image[:, :, 0] - input_image[:, :, 2] > 0.3  # *255
    mask3 = input_image[:, :, 0] > 0.5  # *255

    mask4 = input_image[:, :, 1] > 0.8
    # mask2 = input_image[:,:,2] > 0.5*255
    mask5 = input_image[:, :, 6] < 0.2  # *255
    mask6 = np.logical_or(input_image[:, :, 2] > 0.4, input_image[:, :, 0] < 0.1)  # *255

    mask_combined_1 = np.logical_and(mask1, np.logical_and(mask2, mask3))
    mask_combined_2 = np.logical_and(mask4, np.logical_and(mask5, mask6))
    mask_combined = np.logical_or(mask_combined_1, mask_combined_2)

    # Potential active fires
    mask7 = np.logical_and(
        input_image[:, :, 0] / input_image[:, :, 2] > 1.8,
        input_image[:, :, 0] - input_image[:, :, 2] > 0.17,
    )

    mask_combined_temp = mask7.copy()

    for i in range(mask7.shape[0]):
        for j in range(mask7.shape[1]):
            if mask7[i, j]:

                neighbor_indexes = (
                    slice(
                        max(0, i - neighborhood_size // 2), i + neighborhood_size // 2 + 1
                    ),
                    slice(
                        max(0, j - neighborhood_size // 2), j + neighborhood_size // 2 + 1
                    ),
                )

                neighbor_window = input_image[neighbor_indexes]

                negated_mask = ~np.logical_and(
                    mask7[neighbor_indexes], mask_combined[neighbor_indexes]
                )

                #Image without the unambiguous active fire and potential fires pixels.
                mask_window = neighbor_window * negated_mask[:, :, np.newaxis]

                # Water pixel conditions

                condition_water_1 = np.logical_and(
                    neighbor_window[:, :, 3] > neighbor_window[:, :, 2],
                    np.logical_and(
                        neighbor_window[:, :, 2] > neighbor_window[:, :, 1],
                        neighbor_window[:, :, 6] - neighbor_window[:, :, 0] > 0.2,
                    ),
                )
                condition_water_2 = np.logical_or(
                    neighbor_window[:, :, 4] > neighbor_window[:, :, 5],
                    np.logical_and(
                        neighbor_window[:, :, 4] > neighbor_window[:, :, 5],
                        neighbor_window[:, :, 4] > neighbor_window[:, :, 3],
                    ),
                )

                condition_water_pixels = np.logical_and(
                    condition_water_1, condition_water_2
                )

                condition_water_negated = ~condition_water_pixels

                ### Pixels that are not fires or water pixels
                mask_window = mask_window * condition_water_negated[:, :, np.newaxis]

                condition_1 = input_image[i, j, 0] / input_image[i, j, 2] > np.mean(
                    mask_window[:, :, 0] / mask_window[:, :, 2]
                    + max(0.8, 3 * np.std(mask_window[:, :, 0] / mask_window[:, :, 2]))
                )

                condition_2 = input_image[i, j, 0] / input_image[i, j, 2] > np.mean(
                    mask_window[:, :, 0]
                    + max(
                        0.08,
                        3 * np.std(mask_window[:, :, 0]),
                    )
                )

                condition_3 = input_image[i, j, 0] / input_image[i, j, 1] > 1.6

                real_potential_fire_condition = np.logical_and(
                    condition_1, np.logical_and(condition_2, condition_3)
                )

                if real_potential_fire_condition:
                    mask_combined_temp[i, j] = True
                else:
                    mask_combined_temp[i, j] = False


    mask7 = mask_combined_temp

    mask_combined = np.logical_or(mask_combined, mask7)

    mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255

    return mask_broadcasted

