import os
import re
import pickle
import shutil
import numpy as np

from pathlib import Path


def normalize(band):
    # Function that converts 16bits images into 8bits images
    band_max, band_min = band.max(), band.min()
    return (((band - band_min) / ((band_max - band_min))) * 255).astype(np.uint8)


def normalize_0_1_float(band):
    # Function that converts 16bits images into 8bits images
    band_max, band_min = band.max(), band.min()
    return (((band - band_min) / ((band_max - band_min)))).astype(np.float32)


def normalizeStd(image):
    """Normalize the input around the mean value and twice the standard deviation

    Attributes
    ----------
    image : array
        Image to be normalized with its mean and standard deviation.

    Output
    ------
    Float32 image.

    Notes
    -----
    Applied tipically in visualization purposes.
    """
    return (
        (image - (np.nanmean(image) - np.nanstd(image) * 2))
        / (
            (np.nanmean(image) + np.nanstd(image) * 2)
            - (np.nanmean(image) - np.nanstd(image) * 2)
        )
    ).astype(np.float32)


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


def check_potential_new_events(image_name, source_folder):

    potential_event_condition = False
    confirmed_fire_condition = False

    for _, _, files in os.walk(source_folder):

        for file in files:

            file_name_MSMatch = re.match(r"(.+)_\(", file).group(1)
            if re.match(r"(.+)_\(", image_name).group(1) == file_name_MSMatch:
                confirmed_fire_condition = True

    if not confirmed_fire_condition:
        potential_event_condition = True

    return potential_event_condition


def potsprocess_null_patches(patches_main_folder):

    null_patches_path = os.path.join(patches_main_folder, "Null patches")
    os.makedirs(null_patches_path, exist_ok=True)

    folders_list = ["NIR_SWIR", "RGB", "Vegetation Red", "NIR1"]

    for dir in folders_list:
        os.makedirs(os.path.join(null_patches_path, dir), exist_ok=True)

    Path(os.path.join(null_patches_path, "null_events_list.txt")).touch(exist_ok=True)

    already_classified_condition = False

    patches_removed = []

    for notevent_folder in folders_list:
        notevent_folder_path = os.path.join(patches_main_folder, notevent_folder)
        for notevent_name in os.listdir(notevent_folder_path):
            notevent_path = os.path.join(notevent_folder_path, notevent_name)
            image_name = os.path.basename(notevent_path)

            with open(notevent_path, "rb") as image_file:
                image = pickle.load(image_file)

            if (image.max(), image.min()) == (0, 0):
                with open(
                    os.path.join(null_patches_path, "null_events_list.txt"), "r"
                ) as f:
                    for classified_name in f.readlines():
                        if image_name == classified_name.rstrip("\n"):
                            already_classified_condition = True
                            break
                if not already_classified_condition:

                    with open(
                        os.path.join(null_patches_path, "null_events_list.txt"), "a"
                    ) as txt_file:
                        txt_file.write(f"{image_name}\n")

                    folder_name = re.findall(r"\)_([^\.]+)\.pkl", image_name)[0]
                    if (
                        image_name.replace(f"_{folder_name}.pkl", "")
                        not in patches_removed
                    ):
                        for folder in folders_list:
                            folder_id = folder
                            if folder == "Vegetation Red":
                                folder_id = "VEG"

                            image_folder_name = image_name.replace(
                                folder_name, folder_id
                            )
                            shutil.copyfile(
                                os.path.join(
                                    os.path.dirname(notevent_folder_path),
                                    folder,
                                    image_folder_name,
                                ),
                                os.path.join(
                                    null_patches_path, folder, image_folder_name
                                ),
                            )

                            os.remove(
                                os.path.join(
                                    patches_main_folder, folder, image_folder_name
                                )
                            )  # Delete the file from its original location

                        patches_removed.append(
                            image_name.replace(f"_{folder_name}.pkl", "")
                        )

    print(f"{len(patches_removed)} null patches have been reclassified. ")


def save_as_binary(output_path: str, image: np.ndarray):
    with open(output_path, "wb") as f:
        image.tofile(f)


def read_binary_image(image_path: str, dtype: type = None, shape: list = None):
    with open(image_path, "rb") as f:
        image_bin = f.read()
        if dtype:
            image = np.frombuffer(image_bin, dtype=dtype)
        else:
            image = np.frombuffer(image_bin)

        if shape:
            image = image.reshape((shape))

    return image
