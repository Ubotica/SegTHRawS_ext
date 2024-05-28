#!/bin/bash

# DIR=$(realpath $(dirname "$0"))
# DIR2=$(realpath $(dirname "$DIR"))
# # echo $DIR
# # echo $DIR2

# # if [ -n "$1" ]; then 
# # geo_split=$1
# # echo $geo_split
# # else
# # geo_split=0
# # echo $geo_split
# # fi

# # python3 dataset_script.py --geo_split $geo_split

python3 dataset_script.py
