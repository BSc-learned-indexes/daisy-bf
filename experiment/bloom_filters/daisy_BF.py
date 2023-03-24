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

def choose_number_of_hash_functions(t, F, px, qx):
    if px > t:
        k_x = 0
    elif px <= t and px >= F*t:
        k_x = math.log((t*(1/px)), 2)
    else:
        k_x = math.log(1/F, 2)
    return math.ceil(k_x)
    
def calc_lower_bound(data, F, n, t):
    total = 0
    k_arr = [0]*50
    for _, row in data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        num_k = choose_number_of_hash_functions(t, F, px, qx)
        k_arr[num_k] += 1
        total += px * num_k
    print(k_arr)
    print(f"total: {total}")
    return n * total

def calc_filter_size_from_target_FPR(data, F_target, n, t):
    lower_bound = calc_lower_bound(data, F_target/6, n, t)
    size = math.log(math.e, 2) * lower_bound
    return math.ceil(size)

def normalize_scores(pos_data, neg_data):
    pos_data["px"] = pos_data["score"]
    neg_data["px"] = neg_data["score"]

def get_target_FPR_from_csv(path):
    data = pd.read_csv(path)
    return data['false_positive_rating'].values.tolist()

class Daisy_BloomFilter():
    def __init__(self, t, n, m, F) -> None:
        self.n = n
        self.t = t 
        self.m = m
        self.F = F
        self.hash_functions = []
        self.max_hash_functions = math.ceil(math.log(1/F, 2))
        
        for i in range(self.max_hash_functions):
            self.hash_functions.append(hashfunc(self.m))
        self.table = np.zeros(self.m, dtype=int)

    def insert(self, key, px, qx):
        k_x = choose_number_of_hash_functions(self.n, self.F, px, qx)
        for i in range(k_x):
            table_index = self.hash_functions[i](key)
            self.table[table_index] = 1

    """
    lookup returns:
        False if the element is not in the setself.
        True if there is a posibility of the element being in the set.
    """
    def lookup(self, key, px, qx):
        k_x = choose_number_of_hash_functions(self.n, self.F, px, qx)
        for i in range(k_x):
            table_index = self.hash_functions[i](key)
            if self.table[table_index] == 0:
                return False
        return True

    def test(self, key, px, qx):
        if self.lookup(key, px, qx):
            return 1
        else:
            return 0

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

    mem_arr = []
    mem_wo_model = []
    #FPR_target_arr = get_target_FPR_from_csv(FPR_PATH)
    #FPR_target_arr.reverse()
    FPR_target_arr = []
    t_arr = []
    j = 0.0001
    c = 0.05
    f_i = 0.01
    while j < 1:
        t_arr.append(j + c)
        FPR_target_arr.append(f_i)
        j = 2*j
    FPR_lookup_arr = []

    for t in t_arr:
        

        print(f"Target FPR: {f_i}")
        size = calc_filter_size_from_target_FPR(positive_data, f_i, len(positive_data), t)
        print(f"size: {size}")
        mem_arr.append(size + MODEL_SIZE)
        mem_wo_model.append(size)

        if size != 0:
            daisy = Daisy_BloomFilter(t, len(positive_data), size, f_i)
            for _, row in positive_data.iterrows():
                daisy.insert(row["url"], row["px"], row["qx"])

            num_false_positive = 0
            for _, row in negative_data.iterrows():
                num_false_positive += daisy.test(row["url"], row["px"], row["qx"])

            FPR_lookups = num_false_positive / len(negative_data)
        else:
            FPR_lookups = 1
        print(f"Actual FPR: {FPR_lookups}")
        FPR_lookup_arr.append(FPR_lookups)
        
    print(f"size_arr: {mem_arr}")
    print(f"FPR_target_arr: {FPR_target_arr}")
    print(f"FPR_lookups_arr: {FPR_lookup_arr}")

    data = {"memory": mem_arr, "false_positive_rating": FPR_lookup_arr, "false_positive_target": FPR_target_arr, "t_val": t_arr, "memory_wo_model": mem_wo_model}
    df_data = pd.DataFrame.from_dict(data=data)
    df_data.to_csv(f"{args.out_path}/daisy-BF.csv")
    print(t_arr)