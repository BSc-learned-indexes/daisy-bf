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

    parser.add_argument('--use_Q_dist', action="store", dest="use_Q_dist", type=bool, required=False, default=False, help="uses q_dist for daisy")





    return parser.parse_args()

def load_data():
    if (USE_Q_DIST):
        splitted = DATA_PATH.split(".csv")
        data_path = splitted[0] + "_with_qx.csv"
        print(data_path)
    else:
        data_path = DATA_PATH
    data = pd.read_csv(data_path)
    set_px(data, NORMALIZE_SCORES)
    set_qx(data, CONST_QX, USE_Q_DIST)
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

def lower_bound(positive_data, negative_data, F, n):
    total = 0
    log_k_positive_size_dict = defaultdict(int)
    log_k_negative_size_dict = defaultdict(int)
    for _, row in positive_data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        k = k_hash(F, px, qx, n)
        if k < 0:
            print(f"NEGATIVE VALUE!!! {k} **************************************************************")
        log_k_positive_size_dict[k] += 1
        total += px * k
    print(log_k_positive_size_dict)
    for _, row in negative_data.iterrows():
        px = row["px"]
        qx = row["qx"] 
        k = k_hash(F, px, qx, n)
        if k < 0:
            print(f"NEGATIVE VALUE!!! {k} **************************************************************")
        log_k_negative_size_dict[k] += 1
        total += px * k
    print(log_k_negative_size_dict)

    return n * total, log_k_positive_size_dict, log_k_negative_size_dict

def size(positive_data, negative_data, F_target, n):
    lb, log_k_positive_size_dict, log_k_negative_size_dict = lower_bound(positive_data, negative_data, F_target/6, n)
    print(lb)
    size = math.log(math.e, 2) * lb
    return math.ceil(size), log_k_positive_size_dict, log_k_negative_size_dict

def set_px(data, normalize):
    if normalize:
        score_sum = data["score"].sum()
        print(f"normalizing the scores by dividing with score sum: {score_sum}")
        data["px"] = data["score"].div(score_sum)
    else: 
        data["px"] = data["score"]

def set_qx(data, const_qx, use_Q_dist):
    if const_qx and not use_Q_dist:
        qx = 1 / len(data)
        data["qx"] = qx 
        print(f"qx is set to be constant: {qx}")
    else:
        print("qx is not constant")
        query_sum = data["qx"].sum()
        print(f"normalizing the qx by dividing with qx sum: {query_sum}")
        data["qx"] = data["qx"].div(query_sum)
        # raise Exception(f"Not having a constant qx value is not implemented!")


def get_target_actual_FPR_from_csv(path):
    data = pd.read_csv(path)
    return data['false_positive_rating'].values.tolist()

class Daisy_BloomFilter():
    def __init__(self, n, m, F) -> None:
        self.n = n
        self.m = m
        self.F = F
        self.hash_functions = self.init_hash_functions()
        self.arr = np.zeros(m, dtype=int)
    
    def init_hash_functions(self):
        max_hash_functions = math.ceil(math.log(1/self.F, 2))
        hash_functions = []
        for _ in range(max_hash_functions):
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
        
    def get_actual_FPR(self, data, use_Q_dist):
        k_lookup_dict = defaultdict(int)
        total = 0
        zero_k_total = 0
        for _, row in data.iterrows():
            if use_Q_dist:
                for _ in range(int(row["query_count"])):
                    look_up_val, k_x = self.eval_lookup(row["url"], row["px"], row["qx"])
                    total += look_up_val
                    k_lookup_dict[k_x] += 1
                    if k_x == 0:
                        zero_k_total += 1
            else: 
                look_up_val, k_x = self.eval_lookup(row["url"], row["px"], row["qx"])
                total += look_up_val
                k_lookup_dict[k_x] += 1
                if k_x == 0:
                    zero_k_total += 1
            
        return total / len(data), zero_k_total / len(data), k_lookup_dict
    

    def populate(self, data):
        logger = defaultdict(int)
        for _, row in data.iterrows():
            kx = self.insert(row["url"], row["px"], row["qx"])
            logger[kx] += 1
        print(f"populate hash dist: {logger}")
        return logger


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
    USE_Q_DIST = args.use_Q_dist

    data, positive_data, negative_data = load_data()

    # num_keys = int(len(negative_data)*0.01)
    # print(f"number of keys in the set: {num_keys}")
    # positive_data = positive_data.sample(n=num_keys, random_state=None)
    # all_data = pd.concat([positive_data, negative_data], ignore_index=True)
    # all_data.to_csv("./data/scores/daisy-out.csv")

    mem_result = []
    FPR_targets = get_target_actual_FPR_from_csv(FPR_PATH)
    FPR_result = []
    threshold_values = []
    pct_from_zero_hash_func = []
    bits_set_arr = []
    pct_bits_set_arr = []

    log_k_positive_size_arr = []
    log_k_negative_size_arr = []
    insert_k_arr = []
    lookup_k_arr = []
    
    # FPR_targets = [0.2, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
    # FPR_targets = [0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # FPR_targets = [0.5, 0.4, 0.3, 0 .2, 0.1]
    # FPR_targets = [0.5, 0.4]

    FPR_targets = []
    tmp = 0.5
    while tmp > 10**(-4):
        FPR_targets.append(tmp)
        tmp /= 2
    # FPR_targets.append(10**(-10))
    # FPR_targets.append(10**(-20))
    FPR_targets.append(10**(-30))
    print(FPR_targets)
    # FPR_targets = [0.4]

    for f_i in FPR_targets:
        best_size = None
        closest_FPR = None
        zero_hash_pct = None
        num_bits_set = None
        
        print(f"Target FPR: {f_i}")
        n = len(positive_data)
        filter_size, log_k_positive_size_dict, log_k_negative_size_dict = size(positive_data, negative_data, f_i, n)
        print(f"size: {filter_size}")

        if filter_size > 2:
            daisy = Daisy_BloomFilter(n, filter_size, f_i)
            k_insert_dict = daisy.populate(positive_data)
            bits_set = daisy.arr.sum()
            print(f"sum of 1-bits in the array: {bits_set}")
            print(f"fraction of 1-bits in the array: {bits_set / len(daisy.arr)}")
            actual_FPR, FPR_from_zero_k, k_lookup_dict = daisy.get_actual_FPR(negative_data, USE_Q_DIST)
        else:
            actual_FPR = 1.0
            FPR_from_zero_k = len(negative_data)
            bits_set = 0
            k_lookup_dict = None
            k_insert_dict = None

        print(f"actual_FPR: {actual_FPR}")

        if closest_FPR is None or (abs(closest_FPR - f_i)) > (abs(actual_FPR - f_i)):
            closest_FPR = actual_FPR
            best_size = filter_size
            zero_hash_pct = FPR_from_zero_k
            num_bits_set = bits_set

        FPR_result.append(closest_FPR)
        mem_result.append(best_size)
        pct_from_zero_hash_func.append(zero_hash_pct)
        bits_set_arr.append(num_bits_set)
        pct_bits_set_arr.append(num_bits_set/filter_size)

        log_k_positive_size_dict["FPR_target"] = f_i
        log_k_positive_size_dict["FPR_actual"] = actual_FPR
        log_k_positive_size_dict["size"] = filter_size
        log_k_positive_size_arr.append(log_k_positive_size_dict)

        log_k_negative_size_dict["FPR_target"] = f_i
        log_k_negative_size_dict["FPR_actual"] = actual_FPR
        log_k_negative_size_dict["size"] = filter_size
        log_k_negative_size_arr.append(log_k_negative_size_dict)

        k_insert_dict["FPR_target"] = f_i
        k_insert_dict["FPR_actual"] = actual_FPR
        k_insert_dict["size"] = filter_size
        insert_k_arr.append(k_insert_dict)

        k_lookup_dict["FPR_target"] = f_i
        k_lookup_dict["FPR_actual"] = actual_FPR
        k_lookup_dict["size"] = filter_size
        lookup_k_arr.append(k_lookup_dict)
        print("-----------------------------------")
        

    print(f"size_arr: {mem_result}")
    print(f"FPR_targets: {FPR_targets}")
    print(f"FPR_result: {FPR_result}")

    output = {"memory": mem_result, "false_positive_rating": FPR_result, "false_positive_target": FPR_targets, "FPR_from_zero_k": pct_from_zero_hash_func, "bits_set": bits_set_arr, "pct_ones": pct_bits_set_arr}
    df_out = pd.DataFrame.from_dict(data=output)
    df_out.to_csv(f"{args.out_path}/daisy-BF.csv")

    df_size_positive = pd.DataFrame.from_dict(data=log_k_positive_size_arr)
    df_size_positive.to_csv(f"{args.out_path}/daisy-BF_k_size_positive.csv")

    df_size_negative = pd.DataFrame.from_dict(data=log_k_negative_size_arr)
    df_size_negative.to_csv(f"{args.out_path}/daisy-BF_k_size_negative.csv")

    df_insert = pd.DataFrame.from_dict(data=insert_k_arr)
    df_insert.to_csv(f"{args.out_path}/daisy-BF_k_insert.csv")

    df_lookup = pd.DataFrame.from_dict(data=lookup_k_arr)
    df_lookup.to_csv(f"{args.out_path}/daisy-BF_k_lookup.csv")
