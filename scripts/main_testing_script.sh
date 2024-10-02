#!/bin/bash

# Copyright notice:
# @author Cristopher Castro Traba, Ubotica Technologies
# @copyright 2024 see license file for details


DIR=$(realpath $(dirname "$0"))

MAIN_DIR=$(dirname "$DIR") 

source ${MAIN_DIR}/segthraws_env/bin/activate

# seeds=("0" "1" "2" "42" "73")

# models_list=("unet_smp" "unet++_smp" "deeplabv3plus_smp")
models_list=("small_unet" "small_resunet" "unet3+" "attention_unet")

# encoders_list=("mobilenet_v2" "mobileone_s2" "efficientnet-b0")
# encoder_weights="imagenet"
activation="sigmoid"
lr_list=(0.0003)
# bs_list=(24 48)
loss_list=( "focal_loss_smp" )

min_epochs=400
max_epochs=550

gamma_list=(2)
alpha_list=(0.375)

precision="16-mixed" #16-mixed is accepted in pytorch-lightning 2.2.1

# n_runs_per_seed=3

split="geo"

weakly="_weakly"
lr_scheduler="_lr_scheduler"

# lr_finder=1

n_filters_list=(32)

if [ -z ${n_runs_per_seed+x} ]; then n_runs_per_seed=1; fi

if [ -z ${lr_finder+x} ]; then lr_finder=0; lr_str="_$lr" ; else lr_str="_lr_finder"; lr_list=(0.0003) ; fi

if [ -z ${n_filters_list+x} ]; then n_filters_list=(0); n_filters_str=""; fi

if [[ -z ${encoder_weights+x} ]]; then encoder_weights="none" ; encoder_weights_str="" ; else encoder_weights_str="_$encoder_weights" ; fi

if [[ -z ${encoders_list+x} ]]; then encoders_list=("none") ; encoder_str="" ; fi

models_path="$MAIN_DIR/models"

band_combination_list=("B12 B11 B8A")
# band_combination_list=("B12 B11 B8A" "B04 B03 B02")

for band_list in "${band_combination_list[@]}"; do
    band_list=($band_list)
    band_list_str=$(IFS=_; echo "${band_list[*]}")

	for model in "${models_list[@]}"; do
		for n_filters in "${n_filters_list[@]}"; do
			if [ -z ${n_filters_str+x} ]; then n_filters_str="_$n_filters" ; fi
			for encoder in "${encoders_list[@]}"; do
				if [[ -z ${encoder_str+x} ]]; then encoder_str="_$encoder"; fi
				python3 ${DIR}/get_model_size.py --model_name $model --activation $activation --encoder $encoder --encoder_weights $encoder_weights --n_filters $n_filters
				#### Print the expected size of the model
				for loss in "${loss_list[@]}"; do 

					for lr in "${lr_list[@]}"; do

						for bs in "${bs_list[@]}"; do 

							for alpha in "${alpha_list[@]}"; do   # The quotes are necessary here
								for gamma in "${gamma_list[@]}"; do   # The quotes are necessary here

									model_dir="$OUTPUT_DIR/models/${model}${encoder_str}${encoder_weights_str}${n_filters_str}_fp16_${loss}_gamma${gamma}_alpha${alpha}${lr_str}${lr_scheduler}_${split}${weakly}_${band_list_str}/batch_size_${bs}"

									for seed in "${seeds[@]}"; do   # The quotes are necessary here
										
										#echo -e "${YELLOW} Training the model with seed $seed.${WHITE}"
										echo -e "\nTRAINING THE MODEL $model$encoder_str$n_filters_str WITH LOSS $loss SEED $seed BS $bs LR $lr GAMMA $gamma ALPHA $alpha  BANDS $band_list_str\n"

										n_run=0

										while [ $n_run -lt $n_runs_per_seed ];do

										echo -e "\nMODEL $model$encoder_str TRAINED WITH SEED $seed and BS $bs for  $n_run/$n_runs_per_seed times\n"

										# Main code for training and testing
										python3 ${DIR}/main_testing.py --data_split $split --model_name $model --models_path $models_path --activation $activation --encoder $encoder --loss $loss --gamma $gamma --alpha $alpha --encoder_weights $encoder_weights --bs $bs --min_epoch $min_epochs --max_epoch $max_epochs --seed $seed --precision $precision --lr_finder $lr_finder --band_list ${band_list[@]} --lr $lr --n_filters $n_filters

										n_run=`expr $n_run + 1`
										echo -e "\nMODEL TRAINED WITH SEED $seed for  $n_run/$n_runs_per_seed times\n"

										# Obtain the run file number where the model is saved
										((n_run_save=$(find ${model_dir}/seed_$seed/  -mindepth 1 -maxdepth 1 -type d -name "run*" 2>/dev/null | wc -l)-1))
										# ((n_run_save=$(find ${model_dir}/seed_$seed/  -mindepth 1 -maxdepth 1 -type d -name "run*" | wc -l)))
										echo -e "MODEL SAVED IN FOLDER ${model_dir}/seed_$seed/run_$n_run_save\n"
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done