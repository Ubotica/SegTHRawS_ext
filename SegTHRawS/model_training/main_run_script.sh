#!/bin/bash

DIR=$(realpath $(dirname "$0"))

# source ${DIR}/segthraws_env/bin/activate

OUTPUT_DIR=$(dirname "$(dirname "$DIR")")

# seeds=("2" "3" "4" "5")
seeds=("0" "1" "42")

#models_list=("unet_smp" "unet++_smp" "deeplabv3plus_smp")
models_list=("unet_smp")

# encoders_list=("mobilenet_v2" "mobileone_s2" "efficientnet-b0")
# encoders_list=("mobileone_s2" "efficientnet-b0")
encoders_list=("mobilenet_v2")
# lr=0.0003

activation="sigmoid"

bs_list=(8 16)


loss="focal_loss_smp"
# loss="focal_tversky_loss"
# loss="tversky_loss_smp"
#loss_name="combined_focal_dice_loss_smp"

encoder_weights="imagenet"

min_epochs=150
max_epochs=200

# gamma_list=(2.5 3.0 3.5 4.0)
# alpha_list=(0.5 0.625 0.75 0.9)
alpha_list=(0.9)

precision="16-mixed" #16-mixed is accepted in pytorch-lightning 2.2.1

# n_runs_per_seed=3

split="geo"

##### FOR TESTING

n_runs_per_seed=1
# min_epochs=1
# max_epochs=1
# seeds=(0)
# bs_list=(32)
# # gamma_list=(3 4 5)
# # gamma_list=(2.0 2.5 3.0 4.0)
gamma_list=(2.0)

weakly="_weakly"
lr_scheduler="_lr_scheduler"

for model in "${models_list[@]}"; do

	for encoder in "${encoders_list[@]}"; do

		python3 ${DIR}/get_model_size.py --model_name $model --activation $activation --encoder $encoder --encoder_weights $encoder_weights

		#### Print the expected size of the model

		for bs in "${bs_list[@]}"; do 

			for alpha in "${alpha_list[@]}"; do   # The quotes are necessary here
				for gamma in "${gamma_list[@]}"; do   # The quotes are necessary here
					model_dir="$OUTPUT_DIR/models/${model}_${encoder}_fp16_${loss}_gamma${gamma}_alpha${alpha}${lr_scheduler}_${split}${weakly}/batch_size_${bs}"

					for seed in "${seeds[@]}"; do   # The quotes are necessary here
						
						#echo -e "${YELLOW} Training the model with seed $seed.${WHITE}"
						echo -e "\nTRAINING THE MODEL $model + $encoder WITH SEED $seed BS $bs GAMMA $gamma \n"

						n_run=0

						while [ $n_run -lt $n_runs_per_seed ];do

						echo -e "\nMODEL $model + $encoder TRAINED WITH SEED $seed and BS $bs for  $n_run/$n_runs_per_seed times\n"

						# Main code for training and testing
						python3 ${DIR}/main_run.py --data_split $split --model_name $model --activation $activation --encoder $encoder --loss $loss --gamma $gamma --alpha $alpha --encoder_weights $encoder_weights --bs $bs --min_epoch $min_epochs --max_epoch $max_epochs --seed $seed --precision $precision
						# Main code for training and testing
					#        python3 ${DIR}/main_run.py --model_name $model_name --encoder $encoder --loss $loss_name --encoder_weights $encoder_weights --lr $lr --min_epoch $min_epochs --max_epoch $max_epochs --seed $seed --precision $precision

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
