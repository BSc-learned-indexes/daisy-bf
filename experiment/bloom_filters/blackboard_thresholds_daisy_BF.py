import numpy as np
import pandas as pd
import argparse
from Bloom_filter import hashfunc
import math 
from progress.bar import Bar
import os
import matplotlib.pyplot as plt

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
    parser.add_argument('--model_path', action="store", dest="model_path", type=str, required=True, help="path of the model")
    parser.parse_args()
    return parser.parse_args()
    

def load_data():
    data = pd.read_csv(DATA_PATH)
    positive_sample = data.loc[(data['label']==1)]
    negative_sample = data.loc[(data['label']==-1)]
    return data, positive_sample, negative_sample

def choose_number_of_hash_functions(u, n, F, px, qx, should_print):
    if should_print:
        print("--------------------")
        print("qx: {:.20f}".format(qx))
        print("px: {:.20f}".format(math.log(px, 10)))
        print("px: {:.20f}".format(math.log(px, 2)))
        print("F*px: {:.20f}".format(F*px))
        print("1/n: {:.20f}".format(1/n))
        print("--------------------")
        print("(F/n): {:.20f}".format((F/n)))

    if px > 1/(u*F):
        k_x = 0
    elif 1/u < px < 1/(u*F):
        k_x = math.log(1/(u*F*px), 2)
    elif px < 1/u:
        k_x = math.log(1/F, 2)
    else: 
        raise Exception(f"k could not be calculated from the value: \n n: {n} \n F: {F} \n px {px} \n qx {qx}")
    
    return math.ceil(k_x)
    
def calc_lower_bound(data, F, n, u):
    total = 0
    # print(data.head())
    standard_k = math.log((1/F), 2)
    print(f"log(1/F): {standard_k}")
    k_arr = [0]*50
    should_print = True
    for _, row in data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        # qx = F/n
        qx = 1/n
        # px = 1 - (1/(10**px))
        num_k = choose_number_of_hash_functions(u, n, F, px, qx, should_print)
        should_print = False
        k_arr[num_k] += 1
        # print(num_k, standard_k)
        total += px * num_k
    print(k_arr)
    return n * total

def calc_filter_size_from_target_FPR(data, F_target, n, u):
    lower_bound = calc_lower_bound(data, F_target/6, n, u)
    size = math.log(math.e, 2) * lower_bound
    return math.ceil(size)

def normalize_scores(pos_data, neg_data):
    score_sum = pos_data["score"].sum() + neg_data["score"].sum()
    # print(pos_data.score.unique())
    # print(neg_data.score.unique())
    pos_data["px"] = pos_data["score"]#.div(score_sum)
    neg_data["px"] = neg_data["score"]#.div(score_sum)

def get_target_FPR_from_csv(path):
    data = pd.read_csv(path)
    return data['false_positive_rating'].values.tolist()

if __name__ == '__main__':

    args = init_args()
    DATA_PATH = args.data_path
    FPR_PATH = args.fpr_data_path
    OUT_PATH = args.out_path
    CONST_QX = args.const_qx
    MODEL_SIZE = os.path.getsize(args.model_path) * 8

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


    # size_arr = []
    # f_arr = []
    # f_i =   0.00005
    # max_f = 0.5
    # step =  (max_f - f_i) / 20
    # while f_i < max_f:
    #     print(f"FPR: {f_i}")
    #     size = calc_filter_size_from_target_FPR(positive_data, f_i, len(positive_data))
    #     # size = calc_filter_size_from_target_FPR(pd.concat([positive_data, negative_data]), f_i, len(positive_data))
    #     print(f"size: {size}")
    #     size_arr.append(size / 1000)
    #     f_arr.append(f_i)
    #     f_i += step
    #
    # print(f"size_arr: {size_arr}")
    # print(f"FPR_arr: {f_arr}")
    # mem_arr = []
    # FPR_arr = []
    mem_arr = []
    FPR_arr = get_target_FPR_from_csv(FPR_PATH)
    # FPR_arr = [0.0012668552487721918,
    #         0.0008850632559915312,
    #         0.0008214312571947544,
    #         0.0007404450769079476,
    #         0.0007173061682545742,
    #         0.0006999519867645443,
    #         0.000685490168856186,
    #         0.0006536741694577975,
    #         0.0006392123515494392,]

    for f_i in FPR_arr:
        print(f"FPR: {f_i}")
        size = calc_filter_size_from_target_FPR(positive_data, f_i, len(positive_data), len(all_data))
        # size = calc_filter_size_from_target_FPR(pd.concat([positive_data, negative_data]), f_i, len(positive_data))
        print(f"size: {size}")
        mem_arr.append(size + MODEL_SIZE)
        
    print(f"size_arr: {mem_arr}")
    print(f"FPR_arr: {FPR_arr}")

    data = {"memory": mem_arr, "false_positive_rating": FPR_arr}
    df_data = pd.DataFrame.from_dict(data=data)
    df_data.to_csv(f"{args.out_path}/daisy-BF.csv")
