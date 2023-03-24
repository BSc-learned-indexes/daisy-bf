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
    parser.add_argument('--max_iter', action="store", dest="max_iter", type=int,
                        required=False, help="max iterations to find threshold value", default=10)
    parser.add_argument('--precision', action="store", dest="precision", type=float,
                        required=False, help="minimum precision of actual and target false positive rate", default=0.0001)
    parser.add_argument('--within_ten_pct', action="store", dest="within_ten_pct", type=bool,
                        required=False, help="stop trying to find a new threshold if the current is within 10 \% of the target", default=True)
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
    k_arr = [0]*20
    for _, row in data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        num_k = choose_number_of_hash_functions(t, F, px, qx)
        k_arr[num_k] += 1
        total += px * num_k
    print(k_arr)
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
        
        for _ in range(self.max_hash_functions):
            self.hash_functions.append(hashfunc(self.m))
        self.table = np.zeros(self.m, dtype=int)

    def insert(self, key, px, qx):
        k_x = choose_number_of_hash_functions(self.t, self.F, px, qx)
        for i in range(k_x):
            table_index = self.hash_functions[i](key)
            self.table[table_index] = 1

    """
    lookup returns:
        False if the element is not in the setself.
        True if there is a posibility of the element being in the set.
    """
    def lookup(self, key, px, qx):
        k_x = choose_number_of_hash_functions(self.t, self.F, px, qx)
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
    precision = args.precision
    max_iterations = args.max_iter
    within_ten_pct = args.within_ten_pct

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
    FPR_target_arr = get_target_FPR_from_csv(FPR_PATH)
    FPR_actual_arr = []
    threshold_values = []

    for f_i in FPR_target_arr:
        L = 0
        R = 1
        t = (R + L) / 2
        
        i = 0
        while max_iterations > i:
            t = (R + L) / 2

            print(f"Target FPR: {f_i}")
            alternative_n = len(positive_data[positive_data.px < t])
            n = len(positive_data)
            print(f"n: {n}, a_n: {alternative_n}")
            size = calc_filter_size_from_target_FPR(positive_data, f_i, alternative_n, t)
            print(f"size: {size}")

            if size > 2:
                daisy = Daisy_BloomFilter(t, alternative_n, size, f_i)
                for _, row in positive_data.iterrows():
                    daisy.insert(row["url"], row["px"], row["qx"])

                num_false_positive = 0
                # print(negative_data.head())
                for _, row in negative_data.iterrows():
                    num_false_positive += daisy.test(row["url"], row["px"], row["qx"])

                actual_FPR = num_false_positive / len(negative_data)
            else:
                actual_FPR = 1

            print(f"i: {i}, t: {t}")
            print(f"actual_FPR: {actual_FPR}")

            if actual_FPR < f_i:
                R = t
            else:
                L = t
            
            if abs(actual_FPR - f_i) < precision:
                break
            if abs(actual_FPR - f_i) < 0.1*f_i and within_ten_pct:
                break
            i += 1
            

        FPR_actual_arr.append(actual_FPR)
        mem_arr.append(size)
        threshold_values.append(t)
        print(f"t: {t}")




    print(f"size_arr: {mem_arr}")
    print(f"FPR_target_arr: {FPR_target_arr}")
    print(f"FPR_actual_arr: {FPR_actual_arr}")

    data = {"memory": mem_arr, "false_positive_rating": FPR_actual_arr, "false_positive_target": FPR_target_arr, "t_val": threshold_values}
    df_data = pd.DataFrame.from_dict(data=data)
    df_data.to_csv(f"{args.out_path}/daisy-BF.csv")