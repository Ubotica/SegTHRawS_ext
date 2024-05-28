# SegTHRawS
**PROVISIONAL** This repository corresponds to the Master thesis of Cristopher Castro Traba (cristopher.traba@ubotica.com). The topic of the thesis is: onboard segmentation of thermal hotspots for raw Sentinel-2 multispectral imagery.

The main directory is constituted by a **requirements.txt** file to install the required dependencies for the SegTHRawS environment, and two folders that contain the main code of the project: **SegTHRawS** and **segthraws_module**. They share the same code, but they differ on how the user access it. **SegTHRawS** provides python and shell scripts that can be executed through the terminal, while **segthraws_module** provides a python package to use all the scripts as functions.

## Requirements


## Structure of the code
Inside **SegTHRawS** there are four folders that accounts for the dataset creation, the model training, the model conversion to myriad, and the tiling application. 
### dataset_creation
This folder contains all the available scripts for the dataset generation. If all the requirement specified in [Requirements](#markdown-header-Requirements) are met, the easiest way to generate the main and train datasets is by running **create_datasets.sh**. This shell script expects an argument of 0 or 1, for generating the training dataset with random or geographical split, respectively. If an argument is not passed, the default response of the script is to select the geographical split. 

The next line of code shows how a geographical split can be generated from this shell script:
```
create_datasets.sh 1
```
**dataset_script.py** provides more control in the dataset generation process. If all the requirementes specified in [Requirements](#markdown-header-Requirements) are met, no input is needed, and the default values defined in **constants.py** are used.

The expected input arguments: the path to the original dataset (THRawS in this case), if geographical split wants to be used for the training dataset, the path were the new datasets will be generated, whether weakly segmentation is used for the final segmentation masks, the different splits for the training dataset (train, validation, and testing), and the seed used for the random generators. 

There is an optional argument that defines the path of the dataset that is going to be used to generate the training dataset. If this argument is defined, the training datasets will be generated based on this dataset.

More information on these input arguments can be obtained using the help function in this python script. 

The next line of code shows how to create the main and train dataset with geographical split:
```
python3 SegTHRawS/dataset_creation/dataset_script.py --data_path ../THRawS_data --geo_split 1 --new_dataset_path datasets --weakly 1 --train_split_ratio 0.8 --val_split_ratio 0.1 --test_split_ratio 0.1 -seed 42
```

### model_training

**main_run_script.sh** is the shell script that performs automatic model training for different conditions, without any argument required. All the arguments are defined inside the script and it runs **main_run.py** in the backgorund for a different set of conditions. 

Inside the shell script, the user can specify which models to train, its loss function, if pre-trained weights are used for the encoder, the number of epochs, which seeds and batch sizes, and how many training runs will be performed for each set of conditions. More information on the input arguments can be found in the help function of **main_run.py**.

The expected output from this shell script is a folder called models where the different models trained are included. Each model has associated the pytorch model, the ONNX and UNN model, and its metrics. This script calls the function 'convert_ONNX_to_IR_UNN' inside the **myriad_conversion** folder in the parent directory.

The list of available models, encoders, and loss functions for training can be found in **train_constants.py**. 

