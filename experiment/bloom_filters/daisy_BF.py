import numpy as np
import pandas as pd
import argparse
from Bloom_filter import hashfunc
import math 
import os
from collections import defaultdict

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
                        required=False, help="minimum precision of actual and target false positive rate", default=0.00001)
    parser.add_argument('--WITHIN_TEN_PCT', action="store", dest="within_ten_pct", type=bool,
                        required=False, help="stop trying to find a new threshold if the current is within 10 \% of the target", default=True)
    parser.add_argument('--model_path', action="store", dest="model_path", type=str, required=True, help="path of the model")

    parser.add_argument('--tau', action="store", dest="tau", type=bool, required=False, default=False, help="searches for best threshold value for tau")

    parser.add_argument('--normalize_scores', action="store", dest="normalize", type=bool, required=False, default=False, help="normalizes px and qx scores")


    return parser.parse_args()

def load_data():
    data = pd.read_csv(DATA_PATH)
    positive_sample = data.loc[(data['label']==1)]
    negative_sample = data.loc[(data['label']==-1)]
    return data, positive_sample, negative_sample

def k_hash(F, px, qx, n):
    if qx <= F * px or px > (1 / n): 
        k_x = 0
    elif F * px < qx and qx <= min(px, (F/n)):
        k_x = math.log((1/F)*(qx/px), 2)
    elif qx > px and (F/n) >= px:
        k_x = math.log(1/F, 2)
    elif qx > (F/n) and ((F/n) < px and px <= 1/n):
        k_x = math.log(1/(n*px), 2)
    else: 
        raise Exception(f"k could not be calculated from the value: \n n: {n} \n F: {F} \n px {px} \n qx {qx}")

    return math.ceil(k_x)

def lower_bound(data, F, n):
    total = 0
    logger = defaultdict(int)
    for _, row in data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        k = k_hash(F, px, qx, n)
        logger[k] += 1
        total += px * k
    print(logger)

    return n * total

def size(data, F_target, n):
    lb = lower_bound(data, F_target/6, n)
    print(lb)
    size = math.log(math.e, 2) * lb
    return math.ceil(size)

def normalize_scores(all_data, pos_data, neg_data, normalize):
    if normalize:
        score_sum = pos_data["score"].sum() + neg_data["score"].sum()
        pos_data["px"] = pos_data["score"].div(score_sum)
        neg_data["px"] = neg_data["score"].div(score_sum)
        all_data["px"] = all_data["score"].div(score_sum)
    else: 
        pos_data["px"] = pos_data["score"]
        neg_data["px"] = neg_data["score"]
        all_data["px"] = all_data["score"]

def get_target_actual_FPR_from_csv(path):
    data = pd.read_csv(path)
    return data['false_positive_rating'].values.tolist()

class Daisy_BloomFilter():
    def __init__(self, n, m, F, u) -> None:
        self.n = n
        self.m = m
        self.F = F
        self.hash_functions = self.init_hash_functions()
        self.arr = np.zeros(m, dtype=int)
    
    def init_hash_functions(self):
        max_hash_functions = math.ceil(math.log(1/self.F, 2))
        hash_functions = []
        for i in range(max_hash_functions):
            hash_functions.append(hashfunc(self.m))
        return hash_functions
    

    def insert(self, key, px, qx):
        k_x = k_hash(self.F, px, qx, self.n)
        if k_x == 0:
            return k_x
        for i in range(k_x):
            arr_index = self.hash_functions[i](key)
            self.arr[arr_index] = 1
        return k_x


    """
    lookup returns:
        False if the element is not in the set.
        True if there is a the element is in the set (can make a false positive).
    """
    def lookup(self, key, px, qx):
        k_x = k_hash(self.F, px, qx, self.n)
        for i in range(k_x):
            arr_index = self.hash_functions[i](key)
            if self.arr[arr_index] == 0:
                return False, k_x
        return True, k_x

    def eval_lookup(self, key, px, qx):
        is_in_set, k_x = self.lookup(key, px, qx)
        if is_in_set:
            return 1, k_x
        else:
            return 0, k_x
        
    def get_actual_FPR(self, data):
        total = 0
        zero_k_total = 0
        for _, row in data.iterrows():
            look_up_val, k_x = self.eval_lookup(row["url"], row["px"], row["qx"])
            total += look_up_val
            if k_x == 0:
                zero_k_total += 1
            
        return total / len(data), zero_k_total / len(data)
    

    def populate(self, data):
        logger = defaultdict(int)
        for _, row in data.iterrows():
            kx = self.insert(row["url"], row["px"], row["qx"])
            logger[kx] += 1
        print(f"populate hash dist: {logger}")


if __name__ == '__main__':

    args = init_args()
    DATA_PATH = args.data_path
    FPR_PATH = args.fpr_data_path
    OUT_PATH = args.out_path
    CONST_QX = args.const_qx
    MODEL_SIZE = os.path.getsize(args.model_path) * 8
    PRECISION = args.precision
    MAX_ITERATIONS = args.max_iter
    WITHIN_TEN_PCT = args.within_ten_pct
    TAU = args.tau
    NORMALIZE_SCORES = args.normalize

    all_data, positive_data, negative_data = load_data()

    score_sum = positive_data["score"].sum() + negative_data["score"].sum()
    print(f"sum of all scores: {score_sum}")

    # num_keys = int(len(negative_data)*0.01)
    # print(f"number of keys in the set: {num_keys}")
    # positive_data = positive_data.sample(n=num_keys, random_state=None)
    # all_data = pd.concat([positive_data, negative_data], ignore_index=True)
    # all_data.to_csv("./data/scores/daisy-out.csv")

    # create constant qx:
    qx = 0.0
    if CONST_QX:
        qx = 1 / (len(negative_data) + len(positive_data))
        all_data["qx"] = qx 
        positive_data["qx"] = qx 
        negative_data["qx"] = qx 
        print(f"qx is set to be constant: {qx}")
    
    normalize_scores(all_data, positive_data, negative_data, NORMALIZE_SCORES)


    # WARNING: remember to change this constant!
    B = positive_data["score"].sum() + negative_data["score"].sum()
    print(f"B: {B}")


    mem_result = []
    FPR_targets = get_target_actual_FPR_from_csv(FPR_PATH)
    FPR_result = []
    threshold_values = []
    pct_from_zero_hash_func = []
    bits_set_arr = []
    
    FPR_targets = [0.2, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
    # FPR_targets = [0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # FPR_targets = [0.5, 0.4, 0.3, 0 .2, 0.1]
    # FPR_targets = [0.5, 0.4]

    FPR_targets = []
    tmp = 0.5
    for i in range(7):
        FPR_targets.append(tmp)
        tmp /= 2

    print(FPR_targets)
    FPR_targets = [0.0000001]

    for f_i in FPR_targets:
        best_size = None
        closest_FPR = None
        closest_t = None
        zero_hash_pct = None
        num_bits_set = None
        
        i = 0
        print(f"Target FPR: {f_i}")
        n = len(positive_data)
        filter_size = size(all_data, f_i, n)
        print(f"size: {filter_size}")

        if filter_size > 2:

            daisy = Daisy_BloomFilter(n, filter_size, f_i, len(all_data))
            daisy.populate(positive_data)
            bits_set = daisy.arr.sum()
            print(f"sum of 1-bits in the array: {bits_set}")
            print(f"fraction of 1-bits in the array: {bits_set / len(daisy.arr)}")
            actual_FPR, FPR_from_zero_k = daisy.get_actual_FPR(negative_data)

        else:
            actual_FPR = 1
            FPR_from_zero_k = len(negative_data)
            bits_set = 0

        print(f"i: {i}, t: {TAU_VAL}")
        print(f"actual_FPR: {actual_FPR}")

        if closest_FPR is None or (abs(closest_FPR - f_i)) > (abs(actual_FPR - f_i)):
            closest_FPR = actual_FPR
            best_size = filter_size
            closest_t = TAU_VAL
            zero_hash_pct = FPR_from_zero_k
            num_bits_set = bits_set

        FPR_result.append(closest_FPR)
        mem_result.append(best_size)
        pct_from_zero_hash_func.append(zero_hash_pct)
        threshold_values.append(closest_t)
        bits_set_arr.append(num_bits_set)
        print(f"t: {closest_t}")

    print(f"size_arr: {mem_result}")
    print(f"FPR_targets: {FPR_targets}")
    print(f"FPR_result: {FPR_result}")

    data = {"memory": mem_result, "false_positive_rating": FPR_result, "false_positive_target": FPR_targets, "t_val": threshold_values, "FPR_from_zero_k": pct_from_zero_hash_func, "bits_set": bits_set_arr}
    df_data = pd.DataFrame.from_dict(data=data)
    df_data.to_csv(f"{args.out_path}/daisy-BF.csv")
