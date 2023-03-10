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
    parser.add_argument('--const_qx', action="store", dest="const_qx", type=bool,
                        required=False, help="make qx a constant", default=True)

    parser.parse_args()
    return parser.parse_args()
    

def load_data():
    data = pd.read_csv(DATA_PATH)
    positive_sample = data.loc[(data['label']==1)]
    negative_sample = data.loc[(data['label']==-1)]
    return data, positive_sample, negative_sample

def choose_number_of_hash_functions(n, F, px, qx):
    if qx <= F * px or px > (1 / n): 
        k_x = 0
    elif F * px < qx and qx <= min(px, (F/n)):
        k_x = math.log(1/(F*(qx/px)), 2)
    elif qx > px and (F/n) >= px:
        k_x = math.log(1/F, 2)
    elif qx > (F/n) and ((F/n) < px and px <= 1/n):
        k_x = math.log(1/(n*px), 2)
    else: 
        raise Exception("k could not be calculated.")
    
    return math.ceil(k_x)
    
def calc_lower_bound(data, F, n):
    total = 0
    # print(data.head())
    standard_k = math.log((1/F), 2)
    print(f"log(1/F): {standard_k}")
    k_arr = [0]*10
    for _, row in data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        num_k = choose_number_of_hash_functions(n, F, px, qx)
        k_arr[num_k] += 1
        # print(num_k, standard_k)
        total += px * num_k
    print(k_arr)
    return n * total

def calc_filter_size_from_target_FPR(data, F_target, n):
    lower_bound = calc_lower_bound(data, F_target, n)
    size = math.log(math.e, 2) * lower_bound
    return math.ceil(size)

def normalize_scores(pos_data, neg_data):
    score_sum = pos_data["score"].sum() + neg_data["score"].sum()
    pos_data["px"] = pos_data["score"].div(score_sum)
    neg_data["px"] = neg_data["score"].div(score_sum)

if __name__ == '__main__':

    args = init_args()
    DATA_PATH = args.data_path
    FPR_PATH = args.fpr_data_path
    OUT_PATH = args.out_path
    CONST_QX = args.const_qx

    #Progress bar 
    # bar = Bar('Creating Daisy-BF', max=math.ceil((args.max_size - args.min_size)/args.step))

    all_data, positive_data, negative_data = load_data()

    # create constant qx:
    qx = 0.0
    if CONST_QX:
        qx = 1 / (len(negative_data) + len(positive_data))
        all_data["qx"] = qx 
        positive_data["qx"] = qx 
        negative_data["qx"] = qx 
        print(f"qx is set to be constant: {qx}")
    
    normalize_scores(positive_data, negative_data)

    print(positive_data.head())
    print(negative_data.head())


    size_arr = []
    f_arr = []
    f_i = 0.1
    step = 0.01
    max_f = 0.5
    while f_i < max_f:
        print(f"FPR: {f_i}")
        size = calc_filter_size_from_target_FPR(pd.concat([positive_data, negative_data]), f_i, len(positive_data))
        print(f"size: {size}")
        size_arr.append(size)
        f_arr.append(f_i)
        f_i += step
        
    print(f"size_arr: {size_arr}")
    print(f"FPR_arr: {f_arr}")
    mem_arr = []
    FPR_arr = []
