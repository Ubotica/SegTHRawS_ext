
import os
import argparse

from dataset_creation import dataset_creation_multiprocess
from geo_split_seg_dataset import generate_geo_split_dataset
from random_split_seg_dataset import generate_random_split_dataset


from constants import thraws_data_path, DATASET_PATH

def create_dataset_folders(new_dataset_path: str):
    
    for stage in ['train','val','test']:
        for files in ['images','masks']:
            os.makedirs(os.path.join(new_dataset_path,stage,files),exist_ok=True)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Training dataset generator')


    parser.add_argument('--data_path',          type=str,   help='Path to the raw THRawS data directory',                   default=thraws_data_path)
    parser.add_argument('--geo_split',          type=str,   help='Decide if geographical split is applied. 1: YES; 0: NO',  default=1,choices=[0,1])
    parser.add_argument('--new_dataset_path',   type=str,   help='Path where the new dataset wants to be generated.',       default=None)
    parser.add_argument('--dataset_path',       type=str,   help='Path of the full dataset with all the patches',           default=DATASET_PATH)
    parser.add_argument('--weakly',             type=int,   help='Use weakly supervision. 1: YES; 0: NO ',                  default=1,choices=[0,1])
    parser.add_argument('--band_list',          type=str,   help='Specify the band combination for the dataset creation. Example: ["B12","B11","B8A"] ', nargs='+', default=["B12","B11","B8A"])
    parser.add_argument('--train_split_ratio',  type=float, help='Desired percentage for the training split.',              default=0.8)
    parser.add_argument('--val_split_ratio',    type=float, help='Desired percentage for the validation split.',            default=0.1)
    parser.add_argument('--test_split_ratio',   type=float, help='Desired percentage for the testing split.',               default=0.1)
    parser.add_argument('--seed',               type=int,   help='Seed for random number generators.',                      default=42)
    
    args = parser.parse_args()

    thraws_data_path = args.data_path
    geo_split_condition = bool(args.geo_split)
    new_dataset_path = args.new_dataset_path
    dataset_path = args.dataset_path
    weakly = bool(args.weakly)
    band_list = args.band_list
    train_split_ratio = args.train_split_ratio
    val_split_ratio = args.val_split_ratio
    test_split_ratio = args.test_split_ratio
    seed = args.seed

    band_list_str = "_".join(band_list)

    if geo_split_condition:
        split_str = 'geo'
    else:
        split_str = 'random'

    if weakly:
        dataset_name = f'train_{split_str}_split_weakly_{band_list_str}_dataset'
    else:
        dataset_name = f'train_{split_str}_split_{band_list_str}_dataset'

    if not new_dataset_path:
        new_dataset_path = os.path.join(os.path.dirname(dataset_path),dataset_name)

    # dataset_creation_multiprocess(data_path=thraws_data_path)

    if geo_split_condition:
        print(f"Generating geographical split dataset for bands {band_list_str}")
        generate_geo_split_dataset(new_dataset_path = new_dataset_path,
                                   dataset_path = dataset_path,
                                   weakly = weakly,
                                   band_combination=band_list,
                                   train_split_ratio = train_split_ratio,
                                   val_split_ratio = val_split_ratio,
                                   test_split_ratio = test_split_ratio,
                                   seed = seed,
                                   )
    else:
        print(f"Generating random split dataset for bands {band_list_str}")
        generate_random_split_dataset(new_dataset_path = new_dataset_path,
                                      dataset_path =dataset_path, 
                                      weakly =weakly,
                                      band_combination=band_list,
                                      train_split_ratio =train_split_ratio, 
                                      val_split_ratio =val_split_ratio, 
                                      test_split_ratio =test_split_ratio, 
                                      seed =seed, 
                                      )