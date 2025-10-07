"""
Copyright notice:
@author Cristopher Castro Traba, Ubotica Technologies
@copyright 2024 see license file for details
"""

import os
import re
import random
import pickle
import numpy as np

from copy import deepcopy
from itertools import combinations

from .constants import DATASET_PATH, band_combinations_dict
from ..utils import normalize_0_1_float


def create_dataset_folders(new_dataset_path: str):

    for stage in ["train", "val", "test"]:
        for files in ["images", "masks"]:
            os.makedirs(os.path.join(new_dataset_path, stage, files), exist_ok=True)


def get_n_events_per_scene_dict(dataset_path: str, seed: int = 42) -> dict:
    random.seed(seed)
    masks_paths = os.listdir(os.path.join(dataset_path, "masks", "weakly_segmentation"))

    random.shuffle(masks_paths)

    scene_list = []

    event_scenes_dict = {}

    for mask_name in masks_paths:

        scene_name = re.match(r"(.+)_[0-9]+_G", mask_name).group(1)

        if re.match(r"(.+)_[0-9]+", scene_name):
            scene_name = re.match(r"(.+)_[0-9]+", scene_name).group(1)

        if scene_name not in scene_list:
            number_of_images_per_scene = sum(
                1 for scene in masks_paths if scene.startswith(scene_name)
            )
            scene_list.append(scene_name)
            event_scenes_dict[scene_name] = number_of_images_per_scene

    assert len(masks_paths) == sum(
        event_scenes_dict.values()
    )  # Ensure that all the events were correctly categorized

    return event_scenes_dict


def find_scenes_combination(
    n_elems_per_comb: int = 5, input_dict: dict = None, n_events_max: int = None
):

    for combination in combinations(input_dict.values(), n_elems_per_comb):

        if sum(combination) > n_events_max - 1 and sum(combination) < n_events_max + 1:
            used_values = []
            scenes_combination = []
            for value in combination:
                for key, val in input_dict.items():
                    if val == value and key not in used_values:
                        scenes_combination.append(key)
                        used_values.append(val)
                        break

            return scenes_combination


def geo_split_events_creation(
    dataset_path: str,
    new_dataset_path: str,
    event_scenes_dict: dict,
    scenes_test_combination: list,
    input_band_combination: list = ["B12", "B11", "B8A"],
    normalization: bool = False,
    weakly: bool = False,
    val_split_ratio: float = 0.1,
    seed: int = 42,
):

    random.seed(seed)

    if not dataset_path:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "datasets",
            "main_dataset",
        )

    masks_path = os.path.join(dataset_path, "masks", "weakly_segmentation")

    if not new_dataset_path:
        new_dataset_path = os.path.join(
            os.path.dirname(dataset_path), "train_geo_split_dataset"
        )
        create_dataset_folders(new_dataset_path)

    masks_paths = os.listdir(masks_path)
    random.shuffle(masks_paths)

    events_copy_dict = deepcopy(event_scenes_dict)
    removed_scenes_names = (
        []
    )  # List used for removing the scenes already classified into testing or validation
    val_idx = 0
    val_max_idx = int(np.round(val_split_ratio * sum(event_scenes_dict.values())))

    train_events = 0
    test_events = 0

    for event_mask_name in masks_paths:

        final_image = []
        for band in input_band_combination:
            for band_name, band_values in zip(
                band_combinations_dict.keys(), band_combinations_dict.values()
            ):
                if band in band_values:

                    patch_name = event_mask_name.replace("mask_weakly", band_name)
                    image_path = os.path.join(
                        dataset_path, "images", "event", band_name, patch_name
                    )
                    with open(image_path, "rb") as image_file:
                        image = pickle.load(image_file)

                    final_image.append(image[:, :, band_values.index(band)])
                    break

        image = np.transpose(np.array(final_image), (1, 2, 0))
        if normalization:
            image = normalize_0_1_float(image)
        with open(os.path.join(masks_path, event_mask_name), "rb") as mask_file:
            mask = pickle.load(mask_file)

        if not weakly:
            mask[mask == -1] = 0

        scene_name = re.match(r"(.+)_[0-9]+_G", event_mask_name).group(1)

        patch_name = patch_name.replace(f"_{band_name}", "")

        if re.match(r"(.+)_[0-9]+", scene_name):
            scene_name = re.match(r"(.+)_[0-9]+", scene_name).group(1)

        if scene_name in scenes_test_combination:
            if scene_name not in removed_scenes_names:
                del events_copy_dict[scene_name]
                removed_scenes_names.append(scene_name)

            test_events += 1

            with open(
                os.path.join(
                    new_dataset_path,
                    "test",
                    "masks",
                    event_mask_name.replace("_mask_weakly.pkl", "_mask.bin"),
                ),
                "wb",
            ) as file:
                mask.tofile(file)
            with open(
                os.path.join(
                    new_dataset_path,
                    "test",
                    "images",
                    patch_name.replace(".pkl", ".bin"),
                ),
                "wb",
            ) as file:
                image.tofile(file)

        else:
            if val_idx < val_max_idx:

                val_idx += 1

                with open(
                    os.path.join(
                        new_dataset_path,
                        "val",
                        "masks",
                        event_mask_name.replace("_mask_weakly.pkl", "_mask.bin"),
                    ),
                    "wb",
                ) as file:
                    mask.tofile(file)
                with open(
                    os.path.join(
                        new_dataset_path,
                        "val",
                        "images",
                        patch_name.replace(".pkl", ".bin"),
                    ),
                    "wb",
                ) as file:
                    image.tofile(file)

            else:
                train_events += 1

                with open(
                    os.path.join(
                        new_dataset_path,
                        "train",
                        "masks",
                        event_mask_name.replace("_mask_weakly.pkl", "_mask.bin"),
                    ),
                    "wb",
                ) as file:
                    mask.tofile(file)
                with open(
                    os.path.join(
                        new_dataset_path,
                        "train",
                        "images",
                        patch_name.replace(".pkl", ".bin"),
                    ),
                    "wb",
                ) as file:
                    image.tofile(file)

    print(
        f"    Events generated: {train_events} TRAINING, {val_idx} VAL, and {test_events} TEST "
    )


def geo_split_notevents_creation(
    new_dataset_path: str,
    scenes_test_combination: list,
    input_band_combination: list = ["B12", "B11", "B8A"],
    dataset_path: str = DATASET_PATH,
    train_split_ratio: float = 0.8,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    seed: int = 42,
):
    ### Dataset path corresponds to the original dataset of the images

    # Empty mask for the notevent patches
    mask = np.zeros((256, 256, 1), dtype=np.float32)

    # Intitialice the seed
    random.seed(seed)

    if not new_dataset_path:
        new_dataset_path = os.path.join(
            os.path.dirname(dataset_path), "train_geo_split_dataset"
        )
        create_dataset_folders(new_dataset_path)

    masks_path = os.path.join(dataset_path, "masks", "weakly_segmentation")

    # Obtain the number of different scenes of the dataset

    with open(os.path.join(dataset_path, "granules_completed.txt"), "r") as file:
        available_granules = file.readlines()

    scene_main_names_list = []
    for image_name in available_granules:

        scene_main_name = re.match(r"(.+)_[0-9]+_G", image_name).group(1)
        if re.match(r"(.+)_[0-9]+", scene_main_name):
            scene_main_name = re.match(r"(.+)_[0-9]+", scene_main_name).group(1)

        if scene_main_name not in scene_main_names_list:

            scene_main_names_list.append(scene_main_name)

    n_scenes_train_val = len(scene_main_names_list) - len(
        scenes_test_combination
    )  # Substract the number of scenes selected for testing

    # Define maximum indexes and number of images per scene for training, validation and testing

    train_max_idx = int(np.round(train_split_ratio * len(os.listdir(masks_path))))
    val_max_idx = int(np.round(val_split_ratio * len(os.listdir(masks_path))))
    test_max_idx = int(np.round(test_split_ratio * len(os.listdir(masks_path))))

    max_n_train_images = int(np.round(train_max_idx / n_scenes_train_val))
    max_n_val_images = int(np.round(val_max_idx / n_scenes_train_val))
    max_n_test_images = int(np.round(test_max_idx / len(scenes_test_combination)))

    n_extra_test_images = test_max_idx - max_n_test_images * len(
        scenes_test_combination
    )

    # Initialice loop variables
    val_idx = 0
    test_idx = 0
    train_idx = 0
    train_scene_names_saved = []
    val_scene_names_saved = []
    test_scene_names_saved = []

    extra_test_images = []

    notevents_path = os.path.join(
        dataset_path, "images", "notevent", "NIR_SWIR"
    )  # Select any of the images directories

    notevents_paths = os.listdir(notevents_path)
    random.shuffle(notevents_paths)

    for image_name in notevents_paths:

        scene_main_name = re.match(r"(.+)_[0-9]+_G", image_name).group(1)
        if re.match(
            r"(.+)_[0-9]+", scene_main_name
        ):
            scene_main_name = re.match(r"(.+)_[0-9]+", scene_main_name).group(1)

        if (
            train_idx >= train_max_idx
            and val_idx >= val_max_idx
            and test_idx >= test_max_idx
        ):
            break

        final_image = []
        for band in input_band_combination:
            for band_name, band_values in zip(
                band_combinations_dict.keys(), band_combinations_dict.values()
            ):
                if band in band_values:

                    patch_name = image_name.replace("NIR_SWIR", band_name)
                    image_path = os.path.join(
                        dataset_path, "images", "notevent", band_name, patch_name
                    )
                    with open(image_path, "rb") as image_file:
                        image = pickle.load(image_file)

                    final_image.append(image[:, :, band_values.index(band)])
                    break

        image = np.transpose(np.array(final_image), (1, 2, 0))

        ##### TESTING SPLIT #####

        if scene_main_name in scenes_test_combination:
            if test_scene_names_saved.count(scene_main_name) < max_n_test_images:
                test_scene_names_saved.append(scene_main_name)

                with open(
                    os.path.join(
                        new_dataset_path,
                        "test",
                        "masks",
                        image_name.replace("_NIR_SWIR.pkl", "_mask.bin"),
                    ),
                    "wb",
                ) as file:
                    mask.tofile(file)
                with open(
                    os.path.join(
                        new_dataset_path,
                        "test",
                        "images",
                        image_name.replace("_NIR_SWIR.pkl", ".bin"),
                    ),
                    "wb",
                ) as file:
                    image.tofile(file)

                if test_idx >= test_max_idx:
                    continue
                else:
                    test_idx += 1
            else:
                if len(extra_test_images) < n_extra_test_images:
                    extra_test_images.append(image_name)

        else:

            ##### VALIDATION SPLIT #####

            if val_idx < val_max_idx:
                if val_scene_names_saved.count(scene_main_name) <= max_n_val_images:
                    val_scene_names_saved.append(scene_main_name)

                    with open(
                        os.path.join(
                            new_dataset_path,
                            "val",
                            "masks",
                            image_name.replace("_NIR_SWIR.pkl", "_mask.bin"),
                        ),
                        "wb",
                    ) as file:
                        mask.tofile(file)
                    with open(
                        os.path.join(
                            new_dataset_path,
                            "val",
                            "images",
                            image_name.replace("_NIR_SWIR.pkl", ".bin"),
                        ),
                        "wb",
                    ) as file:
                        image.tofile(file)

                    val_idx += 1
            else:

                ##### TRAINING SPLIT #####
                if train_idx <= train_max_idx:
                    if (
                        train_scene_names_saved.count(scene_main_name)
                        <= max_n_train_images
                    ):
                        train_scene_names_saved.append(scene_main_name)

                        with open(
                            os.path.join(
                                new_dataset_path,
                                "train",
                                "masks",
                                image_name.replace("_NIR_SWIR.pkl", "_mask.bin"),
                            ),
                            "wb",
                        ) as file:
                            mask.tofile(file)
                        with open(
                            os.path.join(
                                new_dataset_path,
                                "train",
                                "images",
                                image_name.replace("_NIR_SWIR.pkl", ".bin"),
                            ),
                            "wb",
                        ) as file:
                            image.tofile(file)

                        train_idx += 1

    if test_idx < test_max_idx:
        ### Not able to get all the testing images from the granules, so new images are included

        for image_name in extra_test_images:
            with open(
                os.path.join(
                    new_dataset_path,
                    "test",
                    "masks",
                    image_name.replace("_NIR_SWIR.pkl", "_mask.bin"),
                ),
                "wb",
            ) as file:
                mask.tofile(file)
            with open(
                os.path.join(
                    new_dataset_path,
                    "test",
                    "images",
                    image_name.replace("_NIR_SWIR.pkl", ".bin"),
                ),
                "wb",
            ) as file:
                image.tofile(file)
            test_idx += 1

    print(
        f"Not events generated: {train_idx} TRAINING, {val_idx} VAL, and {test_idx} TEST "
    )


def generate_geo_split_dataset(
    new_dataset_path: str = None,
    dataset_path: str = DATASET_PATH,
    band_combination: list = ["B12", "B11", "B8A"],
    weakly: int = 0,
    train_split_ratio: float = 0.8,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    seed: int = 42,
):

    if new_dataset_path:
        create_dataset_folders(new_dataset_path)
    else:
        new_dataset_path = os.path.join(
            os.path.dirname(dataset_path), "train_geo_split_dataset"
        )
        create_dataset_folders(new_dataset_path)

    event_scenes_dict = get_n_events_per_scene_dict(
        dataset_path=dataset_path, seed=seed
    )

    # Maximum number of events in tests
    n_events_test_max = int(test_split_ratio * sum(event_scenes_dict.values()))

    scenes_test_combination = find_scenes_combination(
        input_dict=event_scenes_dict, n_events_max=n_events_test_max
    )

    geo_split_events_creation(
        dataset_path=dataset_path,
        new_dataset_path=new_dataset_path,
        input_band_combination=band_combination,
        event_scenes_dict=event_scenes_dict,
        scenes_test_combination=scenes_test_combination,
        weakly=bool(weakly),
        val_split_ratio=val_split_ratio,
        seed=seed,
    )

    geo_split_notevents_creation(
        dataset_path=dataset_path,
        new_dataset_path=new_dataset_path,
        input_band_combination=band_combination,
        scenes_test_combination=scenes_test_combination,
        train_split_ratio=train_split_ratio,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        seed=seed,
    )
