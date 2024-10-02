"""
Copyright notice:
Source code obtained for the Massimetti conditions was obtained from Gabriele Meoni and Roberto del Prete in THRawS (https://arxiv.org/abs/2305.11891), available at: https://github.com/ESA-PhiLab/PyRawS
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import torch
import numpy as np

# [1] Massimetti, Francesco, et al. "Volcanic hot-spot detection using SENTINEL-2:
# a comparison with MODIS–MIROVA thermal data series." Remote Sensing 12.5 (2020): 820."


def get_thresholds(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """It returns the alpha, beta, gamma and S threshold maps for each band as described in [1]

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: alpha threshold map.
        torch.tensor: beta threshold map.
        torch.tensor: S threshold map.
        torch.tensor: gamma threshold map.
    """

    def check_surrounded(img):
        conv = torch.nn.Conv2d(1, 1, 3)
        weight = torch.nn.Parameter(
            torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 1.0]]]]),
            requires_grad=False,
        )
        img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="constant", value=1)
        conv.load_state_dict({"weight": weight, "bias": torch.zeros([1])}, strict=False)

        if sentinel_img.device.type == "cuda":
            conv = conv.cuda()

        with torch.no_grad():
            surrounded = (
                conv(
                    torch.tensor(img_pad, dtype=torch.float32, device=img_pad.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                .squeeze(0)
                .squeeze(0)
            )
            surrounded[surrounded < 8] = 0
            surrounded[surrounded == 8] = 1
            del weight
            del conv
            del img_pad
            torch.cuda.empty_cache()
            return surrounded

    with torch.no_grad():
        alpha = torch.logical_and(
            torch.where(sentinel_img[:, :, 2] >= alpha_thr[2], 1, 0),
            torch.logical_and(
                torch.where(
                    sentinel_img[:, :, 2] / sentinel_img[:, :, 1] >= alpha_thr[0], 1, 0
                ),
                torch.where(
                    sentinel_img[:, :, 2] / sentinel_img[:, :, 0] >= alpha_thr[1], 1, 0
                ),
            ),
        )
        beta = torch.logical_and(
            torch.where(
                sentinel_img[:, :, 1] / sentinel_img[:, :, 0] >= beta_thr[0], 1, 0
            ),
            torch.logical_and(
                torch.where(sentinel_img[:, :, 1] >= beta_thr[1], 1, 0),
                torch.where(sentinel_img[:, :, 2] >= beta_thr[2], 1, 0),
            ),
        )
        S = torch.logical_or(
            torch.logical_and(
                torch.where(sentinel_img[:, :, 2] >= S_thr[0], 1, 0),
                torch.where(sentinel_img[:, :, 0] <= S_thr[1], 1, 0),
            ),
            torch.logical_and(
                torch.where(sentinel_img[:, :, 1] >= S_thr[2], 1, 0),
                torch.where(sentinel_img[:, :, 0] >= S_thr[3], 1, 0),
            ),
        )
        alpha_beta_logical_surrounded = check_surrounded(torch.logical_or(alpha, beta))
        gamma = torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    torch.where(sentinel_img[:, :, 2] >= gamma_thr[0], 1, 0),
                    torch.where(sentinel_img[:, :, 2] >= gamma_thr[1], 1, 0),
                ),
                torch.where(sentinel_img[:, :, 0] >= gamma_thr[2], 1, 0),
            ),
            alpha_beta_logical_surrounded,
        )
    return alpha, beta, S, gamma

def get_alert_matrix_and_thresholds(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """It calculates the alert-matrix for a certain image.

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: alert_matrix threshold map.
        torch.tensor: alpha threshold map.
        torch.tensor: beta threshold map.
        torch.tensor: S threshold map.
        torch.tensor: gamma threshold map.
    """
    with torch.no_grad():
        alpha, beta, S, gamma = get_thresholds(
            sentinel_img, alpha_thr, beta_thr, S_thr, gamma_thr
        )
        alert_matrix = torch.logical_or(
            torch.logical_or(torch.logical_or(alpha, beta), gamma), S
        )
    return alert_matrix, alpha, beta, S, gamma

def cluster_9px(img):
    """It performs the convolution to detect clusters of 9 activate pixels (current pixel and 8 surrounding pixels) are at 1.

    Args:
        img (torch.tensor): input alert-matrix

    Returns:
        torch.tensor: convoluted alert-map
    """

    conv = torch.nn.Conv2d(1, 1, 3)
    weight = torch.nn.Parameter(
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        requires_grad=False,
    )
    img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="constant", value=1)
    if img.device.type == "cuda":
        conv = conv.cuda()
    conv.load_state_dict({"weight": weight, "bias": torch.zeros([1])}, strict=False)
    with torch.no_grad():
        surrounded = (
            conv(torch.tensor(img_pad, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
            .squeeze(0)
            .squeeze(0)
        )
        del weight
        del conv
        del img_pad
        torch.cuda.empty_cache()
    return surrounded

def s2pix_detector(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """Implements first step of the one described in [1] by proving a filtered alert-map.

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: binary classification. It is if at least a cluster of 9 hot pixels is found.
        torch.tensor: filtered alert_matrix threshold map.
        torch.tensor: alert_matrix threshold map.
    """
    with torch.no_grad():
        alert_matrix, _, _, _, _ = get_alert_matrix_and_thresholds(
            sentinel_img, alpha_thr, beta_thr, S_thr, gamma_thr
        )
        filtered_alert_matrix = cluster_9px(alert_matrix)
        filtered_alert_matrix[filtered_alert_matrix < 9] = 0
        filtered_alert_matrix[filtered_alert_matrix == 9] = 1
        return (
            torch.tensor(
                float(torch.any(filtered_alert_matrix != 0)),
                device=filtered_alert_matrix.device,
            ),
            filtered_alert_matrix,
            alert_matrix,
        )

def filter_bbox_list(
    alert_matrix, props, event_bbox_coordinates_list=None, num_pixels_threshold=9
):
    """Filters bounding box lists found in an alert matrix by takking only the bounding boxes having at least
      ""num_pixels_threshold"" active pixels.

    Args:
        alert_matrix (torch.tensor): alert matrix
        props (list): bounding box list.
        event_bbox_coordinates_list (list, optional): bounding box coordinates. Defaults to None.
        num_pixels_threshold (int,9): number of active pixels in a bounding box. Defaults to 9.

    Returns:
        list: filtered bounding box list.
        list: filtered coordinates list.
    """
    bbox_filtered_list = []
    event_bbox_coordinates_filtered_list = []
    if event_bbox_coordinates_list is not None:
        for prop, coords in zip(props, event_bbox_coordinates_list):
            bbox = prop.bbox
            bbox_rounded = [int(np.round(x)) for x in bbox]
            if (
                torch.sum(
                    alert_matrix[
                        bbox_rounded[0] : bbox_rounded[2] + 1,
                        bbox_rounded[1] : bbox_rounded[3] + 1,
                    ]
                )
                >= num_pixels_threshold
            ):
                bbox_filtered_list.append(prop)
                event_bbox_coordinates_filtered_list.append(coords)
    else:
        for prop in props:
            bbox = prop.bbox
            bbox_rounded = [int(np.round(x)) for x in bbox]
            if (
                torch.sum(
                    alert_matrix[
                        bbox_rounded[0] : bbox_rounded[2] + 1,
                        bbox_rounded[1] : bbox_rounded[3] + 1,
                    ]
                )
                >= num_pixels_threshold
            ):
                bbox_filtered_list.append(prop)

    return bbox_filtered_list, event_bbox_coordinates_filtered_list


def Castro_Traba_conditions(input_image: np.ndarray,
                            empty_pixels_pattern: np.ndarray,
                            neighborhood_size: int = 30,
                            ) -> np.ndarray:
    """ Castro Traba conditions algorithm
    
    Attributes
    ----------

    input_image : np.ndarray,
        Input image to apply the conditions for thermal hotspots segmentation
    
    empty_pixel_pattern : np.ndarray,
        Pattern that specify the common area of the multi-band images
    
    neighborhood_size : int,
        Size of the neighborhood used in the contextual step, Default = 30

    Outputs
    -------
    mask_broadcasted : np.ndarray
        Output three-band segmentation mask  

    Notes
    -----

    """
    # The expected band order of the input image is: B12,B11,B8A

    new_image = input_image*empty_pixels_pattern[:,:,np.newaxis]
    new_image = new_image[:,:,::-1]

    #Murphy conditions (Massimetti uses mask2 >= 1.2)
    mask1 =  np.divide(new_image[:,:,2],new_image[:,:,1],out = np.zeros_like(new_image[:,:,0]),where=empty_pixels_pattern==True) >= 1.4
    mask2 =  np.divide(new_image[:,:,2],new_image[:,:,0],out = np.zeros_like(new_image[:,:,0]),where=empty_pixels_pattern==True) >= 1.4
    mask3 =  new_image[:,:,2] >= 0.15#*255



    # mask4 =  new_image[:,:,1]/new_image[:,:,0] >=2
    mask4 =  np.logical_and(new_image[:,:,2] >100/255,new_image[:,:,0] < 50/255)

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
                        np.divide(
                        new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            2
                        ]
                        , new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            1
                        ],
                        out = np.zeros_like(neighbor_window.astype(np.float32)),
                        where = new_image[
                            max(0, i - neighborhood_size // 2) : i
                            + neighborhood_size // 2
                            + 1,
                            max(0, j - neighborhood_size // 2) : j
                            + neighborhood_size // 2
                            + 1,
                            1
                        ]!=False)
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

    # mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255
    mask_broadcasted = mask_combined.astype(np.float32)

    return mask_broadcasted

def Murphy_conditions(input_image: np.ndarray,
                      empty_pixels_pattern: np.ndarray,
                      neighborhood_size: int = 8,
                      ) -> np.ndarray:
    """ Murphy conditions algorithm
    
    Attributes
    ----------

    input_image : np.ndarray,
        Input image to apply the conditions for thermal hotspots segmentation
    
    empty_pixel_pattern : np.ndarray,
        Pattern that specify the common area of the multi-band images
    
    neighborhood_size : int,
        Size of the neighborhood used in the contextual step, Default = 8

    Outputs
    -------
    mask_broadcasted : np.ndarray
        Output three-band segmentation mask  

    Notes
    -----

    """
    # The expected band order of the input image is: B12,B11,B8A
    

    ### Murphy conditions initial fire pixels

    new_image = input_image*empty_pixels_pattern[:,:,np.newaxis]
    new_image = new_image[:,:,::-1]

    #Murphy conditions (Massimetti uses mask2 >= 1.2)
    mask1 =  np.divide(new_image[:,:,2],new_image[:,:,1],out = np.zeros_like(new_image[:,:,0]),where=empty_pixels_pattern==True) >= 1.4
    mask2 =  np.divide(new_image[:,:,2],new_image[:,:,0],out = np.zeros_like(new_image[:,:,0]),where=empty_pixels_pattern==True) >= 1.4
    mask3 =  new_image[:,:,2] >= 0.15#*255

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
                    np.divide(
                    new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        1
                    ]
                    , new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        0
                    ],
                    out = np.zeros_like(neighbor_window.astype(np.float32)),
                    where = new_image[
                        max(0, i - neighborhood_size // 2) : i
                        + neighborhood_size // 2
                        + 1,
                        max(0, j - neighborhood_size // 2) : j
                        + neighborhood_size // 2
                        + 1,
                        0
                    ]!=0)
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

    # mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255
    mask_broadcasted = mask_combined.astype(np.float32)

    return mask_broadcasted

def Massimetti_conditions(input_image: np.ndarray,
                          empty_pixels_pattern: np.ndarray,
                          ) ->np.ndarray:
    """ Massimetti conditions algorithm from PyRawS 
    
    Attributes
    ----------

    input_image : np.ndarray,
        Input image to apply the conditions for thermal hotspots segmentation
    
    empty_pixel_pattern : np.ndarray,
        Pattern that specify the common area of the multi-band images

    Outputs
    -------
    mask_broadcasted : np.ndarray
        Output three-band segmentation mask  

    Notes
    -----

    """
    # The expected band order of the input image is: B12,B11,B8A

    new_image = input_image*empty_pixels_pattern[:,:,np.newaxis]
    input_image_copy = new_image[:,:,::-1].copy() #Needed because s2pix detector expects the 8A band first
    _, raw_filtered_alert_matrix, _ = s2pix_detector(torch.Tensor(input_image_copy))

    # mask_broadcasted = np.stack([raw_filtered_alert_matrix.bool()] * 3, axis=-1).astype(np.uint8)*255
    mask_broadcasted = raw_filtered_alert_matrix.bool().numpy().astype(np.float32)

    return mask_broadcasted

def Kumar_Roy_conditions(input_image: np.ndarray,
                         empty_pixels_pattern: np.ndarray,
                         )->np.ndarray:
    """ Kumar-Roy conditions algorithm
    
    Attributes
    ----------

    input_image : np.ndarray,
        Input image to apply the conditions for thermal hotspots segmentation
    
    empty_pixel_pattern : np.ndarray,
        Pattern that specify the common area of the multi-band images

    Outputs
    -------
    mask_broadcasted : np.ndarray
        Output three-band segmentation mask  

    Notes
    -----

    """
    ### The expected bands are: B12, B11, B8A, B04, B03, B02, and B01

    input_image = input_image*empty_pixels_pattern[:,:,np.newaxis]


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

                        condition_1 = np.divide(input_image[i, j, 0] , input_image[i, j, 2],out=np.array((0),dtype=np.float32),where=input_image[i,j,2]!=0) > np.mean(
                            np.divide(mask_window[:,:,0],mask_window[:,:,2],out=np.zeros_like(mask_window[:,:,0]),where=mask_window[:,:,2]!=0)
                            + max(0.8, 3 * np.std(np.divide(mask_window[:,:,0],mask_window[:,:,2],out=np.zeros_like(mask_window[:,:,0]),where=mask_window[:,:,2]!=0)))
                        )

                        condition_2 = np.divide(input_image[i, j, 0],input_image[i, j, 2],out=np.zeros_like(input_image[i,j,0]),where=input_image[i,j,2]!=0) > np.mean(
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

    # mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255
    mask_broadcasted = mask_combined.astype(np.float32)

    return mask_broadcasted

def Schroeder_conditions(input_image : np.ndarray,
                         empty_pixels_pattern: np.ndarray,
                         neighborhood_size: int = 61,
                         ) -> np.ndarray:
    """ Schroeder conditions algorithm
    
    Attributes
    ----------

    input_image : np.ndarray,
        Input image to apply the conditions for thermal hotspots segmentation
    
    empty_pixel_pattern : np.ndarray,
        Pattern that specify the common area of the multi-band images
    
    neighborhood_size : int,
        Size of the neighborhood used in the contextual step, Default = 61

    Outputs
    -------
    mask_broadcasted : np.ndarray
        Output three-band segmentation mask  

    Notes
    -----

    """
    ### The expected bands are: B12, B11, B8A, B04, B03, B02, and B01
    input_image = input_image*empty_pixels_pattern[:,:,np.newaxis]

    #Unambiguous active fire pixels

    mask1 = np.divide(input_image[:, :, 0], input_image[:, :, 2],out=np.zeros_like(input_image[:,:,2]),where=input_image[:,:,2]!=0) > 2.5
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
        np.divide(input_image[:, :, 0],input_image[:, :, 2],out=np.zeros_like(input_image[:,:,2]),where=input_image[:,:,2]!=0) > 1.8,
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
                    np.divide(neighbor_window[:, :, 3] ,neighbor_window[:, :, 2],out=np.zeros_like(mask_window[:,:,2]),where=neighbor_window[:,:,2]!=0),
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

                condition_1 = np.divide(input_image[i, j, 0] , input_image[i, j, 2],out=np.array((0),dtype=np.float32),where=input_image[i,j,2]!=0) > np.mean(
                    np.divide(mask_window[:, :, 0] , mask_window[:, :, 2],out=np.zeros_like(mask_window[:,:,2]),where=mask_window[:,:,2]!=0)
                    + max(0.8, 3 * np.std(np.divide(mask_window[:, :, 0] , mask_window[:, :, 2],out=np.zeros_like(mask_window[:,:,2]),where=mask_window[:,:,2]!=0)))
                )

                condition_2 = np.divide(input_image[i, j, 0],input_image[i, j, 2],out=np.array((0),dtype=np.float32),where=input_image[i,j,2]!=0) > np.mean(
                    mask_window[:, :, 0]
                    + max(
                        0.08,
                        3 * np.std(mask_window[:, :, 0]),
                    )
                )

                condition_3 = np.divide(input_image[i, j, 0],input_image[i, j, 1],out=np.array((0),dtype=np.float32),where=input_image[i,j,1]!=0) > 1.6

                real_potential_fire_condition = np.logical_and(
                    condition_1, np.logical_and(condition_2, condition_3)
                )

                if real_potential_fire_condition:
                    mask_combined_temp[i, j] = True
                else:
                    mask_combined_temp[i, j] = False


    mask7 = mask_combined_temp

    mask_combined = np.logical_or(mask_combined, mask7)

    # mask_broadcasted = np.stack([mask_combined] * 3, axis=-1).astype(np.uint8)*255
    mask_broadcasted = mask_combined.astype(np.float32)

    return mask_broadcasted
