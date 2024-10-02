#!/bin/bash

# Copyright notice:
# @author Cristopher Castro Traba, Ubotica Technologies
# @copyright 2024 see license file for details

#Create python environment
python3 -m venv segthraws_env

#Activate the environment
source segthraws_env/bin/activate

#Install required packages
pip install -r requirements.txt

#Install segthraws and reduced PyRawS libraries. PyRawS is available at : https://github.com/ESA-PhiLab/PyRawS
pip install -e .


