

![GitHub last commit](https://img.shields.io/github/last-commit/Ubotica/SegTHRawS_ext?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/Ubotica/SegTHRawS_ext?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Ubotica/SegTHRawS_ext?style=flat-square)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

# SegTHRawS

**SHARED EXTERNALLY** This repository corresponds to the Master Thesis of Cristopher Castro Traba (cristopher.traba@ubotica.com) for the Space Engineering Department at Delft University of Technology. As an academic project, it is open to improvements and further development. If you find any bugs, inconsistencies, or have suggestions, please feel free to open an issue.

The Sentinel-2 images are obtained from the [THRawS](https://zenodo.org/records/7908728) (Thermal Hotspots in raw Sentinel-2 data) dataset, which provides a global distribution of raw multispectral images to detect thermal events. Information on how to handle the dataset and process the images is available in its main repository: [PyRawS](https://github.com/ESA-PhiLab/PyRawS).

The Segmentation of Thermal Hotspots in Raw Sentinel-2 data (SegTHRawS) dataset created in this project is available at: [Zenodo](https://zenodo.org/records/14741990).


## Content of the repository
The SegTHRawS repository includes the following directories:

| Directory Name       | Description                                                                                                                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [segthraws](segthraws/)         | Contains the SegTHRawS package with the following subdirectories:                                                                                                                           |
|                      |   1. dataset_creation: Include the code to create the SegTHRawS and the geographical-split segmentation datasets using the majority voting approach and weakly labeling.                        |
|                      |   2. model_training: Includes the code to train and test models with PyTorch Lightning.                                                                                                         |
|                      |   3. fonts: Contains the Charter font used for the figures.                                                                                                                                     |
| [pyraws](pyraws/)               | Contains a reduced version of [PyRawS](https://github.com/ESA-PhiLab/PyRawS), needed for the dataset creation process.                                                               |
| [scripts](scripts/) | Contains scripts and code for creating a segmentation dataset from the THRawS dataset and for training segmentation models with PyTorch Lightning:                                   |
|                      |   1. dataset_script.py: Python script for creating the SEgTHRawS dataset and the segmentation dataset frot training the models.                                                                 |
|                      |   2. train_model_script.py: Python script to train and test segmentation models.                                                                                                                |
|                      |   3. test_model_script.py: Python script to test previously trained segmentation models.                                                                                                        |
|                      |   4. retrain_model_script.py: Python script to retrain previously trained segmentation models.                                                                                                  |
|                      |   5. main_train_script.sh: Shell script to train and test segmentation models by executing train_model_script with different training  configurations.                                          |
|                      |   6. test_model_script.sh: Shell script to test previously trained segmentation models by executing test_model_script with different training  configurations.                                  |
|                      |   7. retrain_model_script.sh: Shell script to retrain segmentation models by executing retrain_model_script with different training  configurations.                                            |
|                      |   8. get_model_size.py: Python script to obtain the model size in MB or Number of parameters.                                                                                                   |




## Installation

Install SegTHRawS executing the Shell script  [build_env](build_env.sh).

The dataset and models path need to be modified accordingly inside [segthraws/main_paths.py](segthraws/main_paths.py). The dataset path must refer to where the THRawS dataset is located. 


## Contacts
Created by Cristopher Castro Traba in collaboration with Delft University of Technology, Ubotica Technologies, and ESA $\Phi$-lab.

* Cristopher Castro Traba - cristopher.traba@ubotica.com
* David Rijlaarsdam - david.rijlaarsdam@ubotica.com 
* Gabriele Meoni - Gabriele.Meoni@esa.int 
* Roberto Del Prete - roberto.delprete@esa.int
* Jian Guo - j.guo@tudelft.nl
