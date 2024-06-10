#!/bin/bash

# band_list=(B12 B11 B8A)
band_combination_list=("B12 B11 B8A" "B04 B03 B02")

for band_list in "${band_combination_list[@]}"; do
    band_list=($band_list)
    band_list_str=$(IFS=_; echo "${band_list[*]}")

    python3 dataset_script.py --band_list ${band_list[@]}
done

