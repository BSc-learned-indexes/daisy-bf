import numpy as np
import pandas as pd
import argparse
from Bloom_filter import hashfunc
import math 
from progress.bar import Bar
import os

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--fpr_data_path', action="store", dest="fpr_data_path", type=str, required=True,
                        help="path of the false positive ratings")
    parser.add_argument('--out_path', action="store", dest="out_path", type=str,
                        required=False, help="path of the output", default="./data/plots/")
    parser.add_argument('--const_q', action="store", dest="const_q", type=bool,
                        required=False, help="make q a constant", default=True)

    parser.parse_args()
    return parser.parse_args()
    

def load_data():
    data = pd.read_csv(DATA_PATH)
    positive_sample = data.loc[(data['label']==1)]
    negative_sample = data.loc[(data['label']==-1)]
    return data, positive_sample, negative_sample

def normalize_scores(pos_data, neg_data):
    k = pos_data["score"].sum() + neg_data["score"].sum()
    # n = len(pos_data["score"]) + len(neg_data)

    norm_neg = neg_data["score"] / k
    norm_pos = pos_data["scores"] / k
    pos_data["norm_score"] = norm_pos
    neg_data["norm_score"] = norm_neg



if __name__ == '__main__':

    args = init_args()
    DATA_PATH = args.data_path
    FPR_PATH = args.fpr_data_path
    OUT_PATH = args.out_path
    CONST_Q = args.const_q

    #Progress bar 
    bar = Bar('Creating Daisy-BF', max=math.ceil((args.max_size - args.min_size)/args.step))

    all_data, positive_data, negative_data = load_data()

    # create constant Q:
    Q = 0.0;
    if CONST_Q:
        Q = 1 / (len(negative_sample) + len(positive_sample))
        print(f"Q is set to be constant: {Q}")

    normalize_scores(positive_data, negative_data)

    mem_arr = []
    FPR_arr = []