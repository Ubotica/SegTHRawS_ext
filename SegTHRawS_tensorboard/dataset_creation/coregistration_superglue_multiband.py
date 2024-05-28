

try:
    from superglue_models.matching import Matching
    from superglue_models.utils import make_matching_plot
except:  # noqa: E722
    # from .superglue_models.matching import Matching

    raise ValueError(
        "SuperGlue model not found. Please, follow the instructions at: "
        + "https://github.com/ESA-PhiLab/PyRawS#set-up-for-coregistration-study."
    )

# try:
#     from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
#     from lightglue.utils import load_image, rbd
# except:  # noqa: E722
#     raise ValueError(
#         "LightGlue model not found. Please, install lightglue: "
#         + "https://github.com/cvg/LightGlue"
#     )


import sys, os

sys.path.insert(1, os.path.join("..", ".."))

from pyraws.utils.visualization_utils import equalize_tensor
import torch
# import kornia
# from kornia.feature import *

import os
import sys
import numpy as np

from pyraws.raw.raw_event import Raw_event

import torch
from torchvision.transforms.functional import rotate


def get_shift_SuperGlue_profiling(
    b0,
    b1,
    n_max_keypoints=1024,
    sinkhorn_iterations=30,
    equalize=True,
    n_std=2,
    device=torch.device("cpu"),
):
    """Get a shift between two bands of a specific event by using SuperGlue.

    Args:
        b0 (torch.tensor): tensor containing band 0 to coregister.
        b1 (torch.tensor): tensor containing band 1 to coregister.
        n_max_keypoints (int, optional): number of max keypoints to match. Defaults to 1024.
        sinkhorn_iterations (int, optional): number of sinkorn iterations. Defaults to 30.
        requested_bands (list): list containing two bands for which perform the study.
        equalize (bool, optional): if True, equalization is performed. Defaults to True.
        n_std (int, optional): Outliers are saturated for equalization at histogram_mean*- n_std * histogram_std.
                               Defaults to 2.
        device (torch.device, optional): torch.device. Defaults to torch.device("cpu").

    Returns:
        float: mean value of the shift.
        torch.tensor: band 0.
        torch.tensor: band 1.
        dict: granule info.
        float: number of matched kyepoints.
    """

    config = {
        "superpoint": {
            "nms_radius": 1,
            "keypoint_threshold": 0.05,
            "max_keypoints": n_max_keypoints,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": 0.9,
        },
    }
    matching = Matching(config).eval().to(device)
    bands = torch.zeros([b0.shape[0], b0.shape[1], 2], device=device)
    bands[:, :, 0] = b0
    bands[:, :, 1] = b1
    if equalize:
        l0_granule_tensor_equalized = equalize_tensor(bands[:, :, :2], n_std)
        b0 = (
            l0_granule_tensor_equalized[:, :, 0]
            / l0_granule_tensor_equalized[:, :, 0].max()
        )
        b1 = (
            l0_granule_tensor_equalized[:, :, 1]
            / l0_granule_tensor_equalized[:, :, 1].max()
        )
    else:
        b0 = bands[:, :, 0] / bands[:, :, 0].max()
        b1 = bands[:, :, 1] / bands[:, :, 1].max()

    pred = matching(
        {"image0": b0.unsqueeze(0).unsqueeze(0), "image1": b1.unsqueeze(0).unsqueeze(0)}
    )
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches, _ = pred["matches0"], pred["matching_scores0"]
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts1) and len(mkpts0):
        # shift = torch.tensor([x - y for (x, y) in zip(mkpts1, mkpts0)], device=device)
        shift = torch.tensor(mkpts1- mkpts0, device=device) #Half the time than line 105
        shift_v, shift_h = shift[:, 0], shift[:, 1]
        shift_v_mean, shift_v_std = torch.mean(shift_v), torch.std(shift_v)
        shift_h_mean, shift_h_std = torch.mean(shift_h), torch.std(shift_h)
        shift_v = shift_v[
            torch.logical_and(
                shift_v > shift_v_mean - shift_v_std,
                shift_v < shift_v_mean + shift_v_std,
            )
        ]
        shift_h = shift_h[
            torch.logical_and(
                shift_h > shift_h_mean - shift_h_std,
                shift_h < shift_h_mean + shift_h_std,
            )
        ]
        shift_mean = torch.round(
            torch.tensor([-shift_h.mean(), -shift_v.mean()], device=device)
        )
    else:
        return [None, None]
    return shift_mean



def SuperGlue_registration(bands_list,event_path,granule_idx,downsampling=True):

    bands_dict = {}
    for band in bands_list:
        if band in ["B12", "B11","B8A", "B07","B06", "B05"]:
            bands_dict[band]=20
        elif band in ["B02", "B03","B04", "B08"]:
            bands_dict[band]=10
        elif band in ["B01","B09","B10"]:
            bands_dict[band]=60
        else:
            raise ValueError(
            "The band ID does not belong to any Sentinel-2 band"
        )     
    raw_event = Raw_event()

    # Read event from data
    raw_event.from_path(  # Path to the event
        raw_dir_path=event_path,
        # Bands to open. Leave to None to use all the bands.
        bands_list=bands_list,
        # If True, verbose mode is on.
        verbose=False,
    )

    raw_granule = raw_event.get_granule(granule_idx).as_tensor()



    bands_shifts = []
    coarse_status = False
    # try:
    #     # for i,(band,resol) in enumerate(bands_dict.items()):
    #     #     first_band = next(iter(bands_dict))
    #     #     if i>0:
    #     #         if band in ["B12", "B11","B10"]:
    #     #             if first_band in ["B12", "B11","B10"]:
    #     #                 bands_shift = get_shift_SuperGlue_profiling(rotate(raw_granule[:,:,0].unsqueeze(2),180).squeeze(2),rotate(raw_granule[:,:,i].unsqueeze(2),180).squeeze(2))
    #     #             else:
    #     #                 bands_shift = get_shift_SuperGlue_profiling(raw_granule[:,:,0],rotate(raw_granule[:,:,i].unsqueeze(2),180).squeeze(2))
    #     #         else:
    #     #             if first_band in ["B12", "B11","B10"]:
    #     #                 bands_shift = get_shift_SuperGlue_profiling(rotate(raw_granule[:,:,0].unsqueeze(2),180).squeeze(2),raw_granule[:,:,i])
    #     #             else:
    #     #                 bands_shift = get_shift_SuperGlue_profiling(raw_granule[:,:,0],raw_granule[:,:,i])



    #     #         # resolution_factor = bands_dict[first_band]/resol
    #     #         resolution_factor = max(bands_dict.values())/resol
    #     #         bands_shifts.append(bands_shift*resolution_factor)


    #     bands_01 = get_shift_SuperGlue_profiling(raw_granule[:,:,0],raw_granule[:,:,1])
    #     bands_12 = get_shift_SuperGlue_profiling(raw_granule[:,:,1],raw_granule[:,:,2])
    #     bands_23 = get_shift_SuperGlue_profiling(raw_granule[:,:,2],raw_granule[:,:,3])
    #     bands_34 = get_shift_SuperGlue_profiling(raw_granule[:,:,3],raw_granule[:,:,4])
    #     bands_45 = get_shift_SuperGlue_profiling(raw_granule[:,:,4],rotate(raw_granule[:,:,5].unsqueeze(2),180).squeeze(2))
    #     # bands_45 = get_shift_SuperGlue_profiling(raw_granule[:,:,4],raw_granule[:,:,5])
    #     bands_56 = get_shift_SuperGlue_profiling(rotate(raw_granule[:,:,5].unsqueeze(2),180).squeeze(2),raw_granule[:,:,6])
    #     bands_67 = get_shift_SuperGlue_profiling(raw_granule[:,:,6],raw_granule[:,:,7])
    #     bands_78 = get_shift_SuperGlue_profiling(raw_granule[:,:,7],raw_granule[:,:,8])
    #     bands_89 = get_shift_SuperGlue_profiling(raw_granule[:,:,8],rotate(raw_granule[:,:,9].unsqueeze(2),180).squeeze(2))

    #     bands_shifts = [np.multiply(bands_01,2),
    #         np.multiply(bands_01,2)+np.multiply(bands_12,2),
    #         np.multiply(bands_01,2)+np.multiply(bands_12,2)+np.multiply(bands_23,2),
    #         bands_01+bands_12+bands_23+bands_34,
    #         bands_01+bands_12+bands_23+bands_34+bands_45,
    #         bands_01+bands_12+bands_23+bands_34+bands_45+bands_56,
    #         bands_01+bands_12+bands_23+bands_34+bands_45+bands_56+bands_67,
    #         bands_01+bands_12+bands_23+bands_34+bands_45+bands_56+bands_67+bands_78,
    #         bands_01+bands_12+bands_23+bands_34+bands_45+bands_56+bands_67+bands_78+bands_89,
    #         ]

    #     raw_coreg_granule = raw_event.coarse_coregistration(  # granule index to fine coregister.
    #         granules_idx=[granule_idx],
    #         # Search for filling elements # among adjacent Raw granules
    #         downsampling=downsampling,
    #         use_complementary_granules=True,
    #         crop_empty_pixels=False,                            ############################# Crop empty disabled to get always the entire image
    #         bands_shifts=bands_shifts
    #     )
    #     coarse_status = False
    # except:

    #     raw_coreg_granule = raw_event.coarse_coregistration(  # granule index to coarse coregister.
    #         granules_idx=[granule_idx],
    #         # Search for filling elements
    #         # among adjacent Raw granules
    #         downsampling=downsampling,
    #         use_complementary_granules=True,
    #         crop_empty_pixels=False,                            ############################# Crop empty disabled to get always the entire image
    #     )
    #     coarse_status = True


    raw_coreg_granule = raw_event.coarse_coregistration(  # granule index to coarse coregister.
        granules_idx=[granule_idx],
        # Search for filling elements
        # among adjacent Raw granules
        downsampling=downsampling,
        use_complementary_granules=True,
        crop_empty_pixels=False,                            ############################# Crop empty disabled to get always the entire image
    )
    coarse_status = True
    
    raw_coordinates = raw_coreg_granule.get_granule_coordinates()


    return raw_coreg_granule,raw_coordinates,coarse_status


