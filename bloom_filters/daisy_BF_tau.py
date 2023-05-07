import numpy as np
import pandas as pd
import argparse
from Bloom_filter import hashfunc
import math 
import os
from collections import defaultdict
print(os.getcwd())

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--fpr_data_path', action="store", dest="fpr_data_path", type=str, required=False,
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

    parser.add_argument('--Q_dist', action="store", dest="Q_dist", type=bool, required=False, default=False, help="uses q_dist for daisy")
    return parser.parse_args()

TAU = 1

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

    total = 0
    total_2 = 0
    for row in data.itertuples():
        total += row.px * row.qx
        total_2 += row.px * (1/len(data))
    print(f"sum(px*qx): {total}, n: {len(positive_sample)}")
    print(f"sum(px*qx) <= F/n, when F=0.0001: {total} <= {0.0001/len(positive_sample)}")
    # print(f"sum(px*qx): {total}, n: {len(positive_sample)}")
    # print(f"sum(px*1/u): {total_2}, n: {len(positive_sample)}")
    # print(f"sum(px*qx) * n = {total * len(positive_sample)} <= F")
    # print(f"sum(px*1/u) * n = {total_2 * len(positive_sample)} <= F")
    # print(f"|u|: {len(data)}")
    # print(f"n: {len(positive_sample)}")
    print(f"|u|/n = {len(data)/len(positive_sample)} >= F") 

    return data, positive_sample, negative_sample

def k_hash(F, px, qx, n):
    if px >= (1/F) * TAU:
        k_x = 0
    elif TAU <= px <= (1/F) * TAU:
        k_x = math.log(TAU/(F*px), 2)
    elif px < TAU:
        k_x = math.log(1/F, 2)
    else:
        raise Exception(f"k could not be calculated from the value: \n n: {n} \n F: {F} \n px {px} \n qx {qx}")
    # if qx <= F * px or px > (1 / n): 
    #     k_x = 0
    # elif F * px < qx and qx <= min(px, (F/n)):
    #     k_x = math.log((1/F)*(qx/px), 2)
    # elif qx > px and (F/n) >= px:
    #     k_x = math.log(1/F, 2)
    # elif qx > (F/n) and ((F/n) < px and px <= 1/n):
    #     k_x = math.log(1/(n*px), 2)
    # else: 
    #     raise Exception(f"k could not be calculated from the value: \n n: {n} \n F: {F} \n px {px} \n qx {qx}")

    return math.ceil(k_x)

def lower_bound(positive_data, negative_data, F, n):
    total = 0
    for row in positive_data.itertuples(index=False):
        k = k_hash(F, row.px, row.qx, n)
        total += row.px * k

    for row in negative_data.itertuples(index=False):
        k = k_hash(F, row.px, row.qx, n)
        total += row.px * k

    return n * total

def size(positive_data, negative_data, F_target, n):
    lb = lower_bound(positive_data, negative_data, F_target, n)
    print(lb)
    size = math.log(math.e, 2) * lb
    return math.ceil(size)

def set_px(data, normalize):
    if normalize:
        score_sum = data["score"].sum()
        print(f"normalizing the scores by dividing with score sum: {score_sum}")
        data["px"] = data["score"].div(score_sum)
    else: 
        data["px"] = data["score"]

def set_qx(data, const_qx, Q_dist):
    if const_qx and not Q_dist:
        qx = 1 / len(data)
        data["qx"] = qx 
        print(f"qx is set to be constant: {qx}")
    else:
        print("qx is not constant")
        query_sum = data["qx"].sum()
        print(f"normalizing the qx by dividing with qx sum: {query_sum}")
        data["qx"] = data["qx"].div(query_sum)
        print(f"sum of qx is: {data['qx'].sum()}")

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
        
    def get_actual_FPR(self, data, Q_dist):
        n_queries = 0
        k_lookup_dict = defaultdict(int)
        from_model = 0
        total = 0
        if Q_dist:
            for row in data.itertuples(index=False):
                n_queries += row.query_count
                look_up_val, k_x = self.eval_lookup(row.url, row.px, row.qx)
                total += look_up_val * row.query_count
                k_lookup_dict[k_x] += row.query_count
        else:
            n_queries = len(data)
            for row in data.itertuples(index=False):
                look_up_val, k_x = self.eval_lookup(row.url, row.px, row.qx)
                total += look_up_val
                k_lookup_dict[k_x] += 1
        return total / n_queries, k_lookup_dict[0] / n_queries, k_lookup_dict, total-k_lookup_dict[0]
    

    def populate(self, data):
        logger = defaultdict(int)
        for row in data.itertuples(index=False):
            kx = self.insert(row.url, row.px, row.qx)
            logger[kx] += 1
        print(f"populate hash dist: {logger}")
        return logger


if __name__ == '__main__':

    args = init_args()
    DATA_PATH = args.data_path
    FPR_PATH = args.fpr_data_path
    OUT_PATH = args.out_path
    CONST_QX = args.const_qx
    # MODEL_SIZE = os.path.getsize(args.model_path) * 8
    PRECISION = args.precision
    MAX_ITERATIONS = args.max_iter
    WITHIN_TEN_PCT = args.within_ten_pct
    # TAU = args.tau
    NORMALIZE_SCORES = args.normalize
    USE_Q_DIST = args.Q_dist

    data, positive_data, negative_data = load_data()

    # num_keys = int(len(negative_data)*0.01)
    # print(f"number of keys in the set: {num_keys}")
    # positive_data = positive_data.sample(n=num_keys, random_state=None)
    # all_data = pd.concat([positive_data, negative_data], ignore_index=True)
    # all_data.to_csv("./data/scores/daisy-out.csv")

    mem_result = []
    # FPR_targets = get_target_actual_FPR_from_csv(FPR_PATH)
    FPR_result = []
    threshold_values = []
    pct_from_zero_hash_func = []
    bits_set_arr = []
    pct_bits_set_arr = []

    insert_k_arr = []
    lookup_k_arr = []
    model_FP = []
    bloom_FP = []
    
    FPR_targets = []
    tmp = 0.5
    while tmp > 10**(-8):
        FPR_targets.append(tmp)
        tmp /= 2
    # FPR_targets.append(10**(-30))
    print(FPR_targets)

    for f_i in FPR_targets:
        f_i = f_i / 6
        best_size = None
        closest_FPR = None
        zero_hash_pct = None
        num_bits_set = None
        best_fp_bloom = None
        best_lookup_dict = None
        best_k_insert_dict = None

        L = 0
        R = 1
        TAU = (R + L) / 2

        best_size = None
        closest_FPR = None
        closest_t = None
        
        i = 0
        while MAX_ITERATIONS > i:
            TAU = (R + L) / 2
        
            print(f"Target FPR: {f_i}")
            n = len(positive_data)
            filter_size = size(positive_data, negative_data, f_i, n)
            print(f"size: {filter_size}")
            daisy = Daisy_BloomFilter(n, filter_size, f_i)
            k_insert_dict = daisy.populate(positive_data)
            bits_set = daisy.arr.sum()
            print(f"sum of 1-bits in the array: {bits_set}")
            print(f"fraction of 1-bits in the array: {bits_set / len(daisy.arr)}")
            actual_FPR, FPR_from_zero_k, k_lookup_dict, fp_bloom = daisy.get_actual_FPR(negative_data, USE_Q_DIST)

            print(f"actual_FPR: {actual_FPR}")

            if closest_FPR is None or (abs(closest_FPR - f_i)) > (abs(actual_FPR - f_i)):
                closest_FPR = actual_FPR
                best_size = filter_size
                zero_hash_pct = FPR_from_zero_k
                num_bits_set = bits_set 
                best_fp_bloom = fp_bloom
                best_lookup_dict = k_lookup_dict
                best_k_insert_dict = k_insert_dict

            if actual_FPR < f_i:
                R = TAU
            else:
                L = TAU
            i += 1

        bloom_FP.append(best_fp_bloom)
        model_FP.append(best_k_lookup_dict[0])
        FPR_result.append(closest_FPR)
        mem_result.append(best_size)
        pct_from_zero_hash_func.append(zero_hash_pct)
        bits_set_arr.append(num_bits_set)
        pct_bits_set_arr.append(num_bits_set/filter_size)

        k_insert_dict["FPR_target"] = f_i
        k_insert_dict["FPR_actual"] = closest_FPR
        k_insert_dict["size"] = best_size
        insert_k_arr.append(best_k_insert_dict)

        k_lookup_dict["FPR_target"] = f_i
        k_lookup_dict["FPR_actual"] = closest_FPR
        k_lookup_dict["size"] = best_size
        lookup_k_arr.append(best_k_lookup_dict)


        print("-----------------------------------")
        

    print(f"size_arr: {mem_result}")
    print(f"FPR_targets: {FPR_targets}")
    print(f"FPR_result: {FPR_result}")

    output = {"size": mem_result, "false_positive_rating": FPR_result, "false_positive_target": FPR_targets, "FPR_from_zero_k": pct_from_zero_hash_func, "bits_set": bits_set_arr, "pct_ones": pct_bits_set_arr, "model_FP": model_FP, "bloom_FP": bloom_FP}
    df_out = pd.DataFrame.from_dict(data=output)
    df_out.to_csv(f"{args.out_path}/daisy-BF.csv")

    df_insert = pd.DataFrame.from_dict(data=insert_k_arr)
    df_insert.to_csv(f"{args.out_path}/daisy-BF_k_insert.csv")

    df_lookup = pd.DataFrame.from_dict(data=lookup_k_arr)
    df_lookup.to_csv(f"{args.out_path}/daisy-BF_k_lookup.csv")
