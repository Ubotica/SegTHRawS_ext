import os
import sys

# sys.path.insert(1,'..')
sys.path.insert(1,os.path.dirname(os.path.dirname(__file__)))

from segthraws.dataset_creation.dataset_creation import dataset_creation_multiprocess
from segthraws.dataset_creation.geo_split_seg_dataset import generate_geo_split_dataset
from segthraws.dataset_creation.random_split_seg_dataset import generate_random_split_dataset

import argparse

from segthraws.dataset_creation.constants import thraws_data_path

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Training dataset generator')


    parser.add_argument('--data_path',          type=str,   help='Path to the raw THRawS data directory',                   default=thraws_data_path)
    parser.add_argument('--geo_split',          type=str,   help='Decide if geographical split is applied. 1: YES; 0: NO',  default=1,choices=[0,1])
    parser.add_argument('--new_dataset_path',   type=str,   help='Path where the new dataset wants to be generated.',       default=None)
    parser.add_argument('--dataset_path',       type=str,   help='Path of the full dataset with all the patches',           default=None)
    parser.add_argument('--weakly',             type=int,   help='Use weakly supervision. 1: YES; 0: NO ',                  default=0,choices=[0,1])
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
    train_split_ratio = args.train_split_ratio
    val_split_ratio = args.val_split_ratio
    test_split_ratio = args.test_split_ratio
    seed = args.seed


    dataset_creation_multiprocess(data_path=thraws_data_path)

    if geo_split_condition:
        generate_geo_split_dataset(new_dataset_path = new_dataset_path,
                                dataset_path = dataset_path,
                                weakly = weakly,
                                train_split_ratio = train_split_ratio,
                                val_split_ratio = val_split_ratio,
                                test_split_ratio = test_split_ratio,
                                seed = seed,
                                )
    else:
        generate_random_split_dataset(new_dataset_path = new_dataset_path,
                                    dataset_path =dataset_path, 
                                    weakly =weakly, 
                                    train_split_ratio =train_split_ratio, 
                                    val_split_ratio =val_split_ratio, 
                                    test_split_ratio =test_split_ratio, 
                                    seed =seed, 
                                    )