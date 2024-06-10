#!/bin/bash

# DIR=$(realpath $(dirname "$0"))
# DIR2=$(realpath $(dirname "$DIR"))
# echo $DIR
# echo $DIR2

# if [ -n "$1" ]; then 
# geo_split=$1
# echo $geo_split
# else
# geo_split=0
# echo $geo_split
# fi

#python3 dataset_creation.py

# band_list=(B12 B11 B8A)
band_combination_list=("B12 B11 B8A" "B04 B03 B02")

for band_list in "${band_combination_list[@]}"; do
    band_list=($band_list)
    band_list_str=$(IFS=_; echo "${band_list[*]}")

    python3 dataset_script.py --band_list ${band_list[@]}
    # echo ${band_list[@]}
done

