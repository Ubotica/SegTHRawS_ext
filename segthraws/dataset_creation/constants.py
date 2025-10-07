"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
from .threshold_conditions import (
    Castro_Traba_conditions,
    Massimetti_conditions,
    Murphy_conditions,
    Schroeder_conditions,
    Kumar_Roy_conditions,
)

from ..main_paths import thraws_data_path, DATASETS_PATH

SEGTHRAWS_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
MAIN_DIRECTORY = os.path.dirname(SEGTHRAWS_DIRECTORY)

if DATASETS_PATH == "":
    DATASETS_PATH = os.path.join(os.path.dirname(thraws_data_path), "datasets")

DATASET_PATH = os.path.join(DATASETS_PATH, "main_dataset")
os.makedirs(DATASET_PATH, exist_ok=True)

PATCH_SIZE = 256  # Determines size of the dataset images
PATCH_STEP = 192  # This ensures an overlap of 25% in each direction

bands_list = ["B02", "B08", "B03", "B04", "B05", "B11", "B06", "B07", "B8A", "B12"]
# bands_list = [ "B8A", "B11", "B12","B04", "B03", "B02","B08", "B07", "B06","B05"]

directories = {
    "events_images_NIR_SWIR_path": ("images", "event", "NIR_SWIR"),
    "notevents_images_NIR_SWIR_path": ("images", "notevent", "NIR_SWIR"),
    "potential_events_images_NIR_SWIR_path": ("images", "potential_event", "NIR_SWIR"),
    "events_images_RGB_path": ("images", "event", "RGB"),
    "notevents_images_RGB_path": ("images", "notevent", "RGB"),
    "potential_events_images_RGB_path": ("images", "potential_event", "RGB"),
    "events_images_VNIR_path": ("images", "event", "VNIR"),
    "notevents_images_VNIR_path": ("images", "notevent", "VNIR"),
    "potential_events_images_VNIR_path": ("images", "potential_event", "VNIR"),
    "events_images_NIR1_path": ("images", "event", "NIR1"),
    "notevents_images_NIR1_path": ("images", "notevent", "NIR1"),
    "potential_events_images_NIR1_path": ("images", "potential_event", "NIR1"),
    "Castro_Traba_events_path": ("masks", "event", "Castro-Traba"),
    "Castro_Traba_potential_events_path": ("masks", "potential_event", "Castro-Traba"),
    "Massimetti_masks_path": ("masks", "event", "Massimetti"),
    "Massimetti_potential_masks_path": ("masks", "potential_event", "Massimetti"),
    "Murphy_masks_path": ("masks", "event", "Murphy"),
    "Murphy_potential_masks_path": ("masks", "potential_event", "Murphy"),
    "Schroeder_masks_path": ("masks", "event", "Schroeder"),
    "Schroeder_potential_masks_path": ("masks", "potential_event", "Schroeder"),
    "Kumar_Roy_masks_path": ("masks", "event", "Kumar-Roy"),
    "Kumar_Roy_potential_masks_path": ("masks", "potential_event", "Kumar-Roy"),
    "masks_event_voting_2_path": ("masks", "event", "comparison", "voting_2"),
    "masks_event_voting_3_path": ("masks", "event", "comparison", "voting_3"),
    "masks_event_voting_4_path": ("masks", "event", "comparison", "voting_4"),
    "masks_event_intersection_path": ("masks", "event", "comparison", "intersection"),
    "masks_potential_event_voting_2_path": (
        "masks",
        "potential_event",
        "comparison",
        "voting_2",
    ),
    "masks_potential_event_voting_3_path": (
        "masks",
        "potential_event",
        "comparison",
        "voting_3",
    ),
    "masks_comparison_events_plot_voting_4_path": (
        "masks",
        "event",
        "comparison",
        "comparison_plot",
        "voting_4",
    ),
    "masks_comparison_events_plot_intersection_path": (
        "masks",
        "event",
        "comparison",
        "comparison_plot",
        "intersection",
    ),
    "masks_comparison_potential_events_plot_voting_2_path": (
        "masks",
        "potential_event",
        "comparison",
        "comparison_plot",
        "voting_2",
    ),
    "masks_comparison_potential_events_plot_voting_3_path": (
        "masks",
        "potential_event",
        "comparison",
        "comparison_plot",
        "voting_3",
    ),
    "masks_weakly_segmentation_path": ("masks", "weakly_segmentation"),
}

# # Create directories
keys = list(directories.keys())

paths_dict = {key: None for key in keys}
for i, directory in enumerate(directories.values()):
    path = os.path.join(DATASET_PATH, *directory)

    paths_dict[f"{keys[i]}"] = path
    os.makedirs(path, exist_ok=True)


masks_events_dirs = [
    paths_dict["Castro_Traba_events_path"],
    paths_dict["Massimetti_masks_path"],
    paths_dict["Murphy_masks_path"],
    paths_dict["Schroeder_masks_path"],
    paths_dict["Kumar_Roy_masks_path"],
]
masks_potential_events_dirs = [
    paths_dict["Castro_Traba_potential_events_path"],
    paths_dict["Massimetti_potential_masks_path"],
    paths_dict["Murphy_potential_masks_path"],
    paths_dict["Schroeder_potential_masks_path"],
    paths_dict["Kumar_Roy_potential_masks_path"],
]


plot_names = [
    "NIR_SWIR image",
    "Castro-Traba",
    "Massimetti",
    "Murphy",
    "Schroeder",
    "Kumar-Roy",
    "Voting 2",
    "Voting 3",
    " Voting 4",
    "Intersection",
]

mask_generation_functions = (
    Castro_Traba_conditions,
    Massimetti_conditions,
    Murphy_conditions,
    Schroeder_conditions,
    Kumar_Roy_conditions,
)

band_combinations_dict = {
    "NIR_SWIR": ["B12", "B11", "B8A"],
    "RGB": ["B04", "B03", "B02"],
    "VNIR": ["B07", "B06", "B05"],
}
