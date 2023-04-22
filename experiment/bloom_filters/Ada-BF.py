import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from Bloom_filter import hashfunc
import math 
from progress.bar import Bar
# import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True,
                    help="Minimum number of groups")
parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True,
                    help="Maximum number of groups")
parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=False, default=2.0,
                    help="minimum ratio of the keys")
parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=False, default=3.0,
                    help="maximum ratio of the keys")
parser.add_argument('--model_path', action="store", dest="model_path", type=str, required=True,
                    help="path of the model")

parser.add_argument('--min_size', action="store", dest="min_size", type=int, default = 100_000,
                    help="minimum memory to allocate for model + bloom filters")
parser.add_argument('--max_size', action="store", dest="max_size", type=int, default = 500_000,
                    help="maximum memory to allocate for model + bloom filters")
parser.add_argument('--step', action="store", dest="step", type=int, default = 50_000,
                    help="amount of memory to step for each")
parser.add_argument('--out_path', action="store", dest="out_path", type=str,
                    required=False, help="path of the output", default="./data/plots/")




args = parser.parse_args()
DATA_PATH = args.data_path
num_group_min = args.min_group
num_group_max = args.max_group
c_min = args.c_min
c_max = args.c_max
# model_size = os.path.getsize(args.model_path) * 8 #get it to bits instead of byte
model_size = 0



# DATA_PATH = './URL_data.csv'
# num_group_min = 8
# num_group_max = 12
# R_sum = 200000
# c_min = 1.8
# c_max = 2.1


'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
negative_sample = data.loc[(data['label']==-1)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)

class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = []
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
    def insert(self, key, k):
        for j in range(int(k)):
            t = self.h[j](key)
            self.table[t] = 1
    def test(self, key, k):
        test_result = 0
        match = 0
        for j in range(int(k)):
            t = self.h[j](key)
            match += 1*(self.table[t] == 1)
        if match == k:
            test_result = 1
        return test_result



def R_size(count_key, count_nonkey, R0):
    R = [0]*len(count_key)
    R[0] = R0
    for k in range(1, len(count_key)):
        R[k] = max(int(count_key[k] * (np.log(count_nonkey[0]/count_nonkey[k])/np.log(0.618) + R[0]/count_key[0])), 1)
    return R


def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]
    bloom_filter_opt = None

    k_min = 0
    for k_max in range(num_group_min, num_group_max+1):
        for c in c_set:
            tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = np.zeros(k_max - k_min + 1)
            thresholds[-1] = 1.1
            num_negative = sum(train_negative['score'] <= thresholds[-1])
            num_piece = int(num_negative / tau) + 1
            score = train_negative.loc[(train_negative['score'] <= thresholds[-1]), 'score']
            score = np.sort(score)
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = score[score < thresholds[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]

            url = positive_sample['url']
            score = positive_sample['score']

            for score_s, url_s in zip(score, url):
                ix = min(np.where(score_s < thresholds)[0])
                k = k_max - ix
                bloom_filter.insert(url_s, k)

            url_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'url']
            score_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'score']

            ### False positives from the model:
            ML_positive = len(train_negative[train_negative["score"] >=thresholds[-2]])

            test_result = 0
            for score_s, url_s in zip(score_negative, url_negative):
                ix = min(np.where(score_s < thresholds)[0])
                # thres = thresholds[ix]
                k = k_max - ix
                test_result += bloom_filter.test(url_s, k)
            FP_items = test_result + ML_positive

            print('False positive items: %d, Number of groups: %d, c = %f' %(FP_items, k_max, round(c, 2)))

            if FP_opt > FP_items or bloom_filter_opt is None:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max
                opt_c = c

    # print('Optimal FPs: %f, Optimal c: %f, Optimal num_group: %d' % (FP_opt, c_opt, num_group_opt))
    return bloom_filter_opt, thresholds_opt, k_max_opt, opt_c



'''
Implement Ada-BF
'''
if __name__ == '__main__':


        #Progress bar 
    bar = Bar('Creating Ada-BF', max=math.ceil((args.max_size - args.min_size)/args.step))


    mem_arr = []
    FPR_arr = []
    num_regions_arr = []
    c_arr = []

    region_negatives_arr = []
    region_positives_arr = []

    

    i = args.min_size
    while i <= args.max_size:
        print(f"current memory allocated: {i}")
        '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
        bloom_filter_opt, thresholds_opt, k_max_opt, c_opt = Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, (i-model_size), train_negative, positive_sample)



        '''Stage 2: Run Ada-BF on all the samples'''
        ### Test URLs
        url_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'url']
        score_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'score']

        ### False positives from the model:
        ML_positive = len(negative_sample[negative_sample["score"] >=thresholds_opt[-2]])

        test_result = 0
        lookup_negative_logger_dict = defaultdict(int)

        for score_s, url_s in zip(score_negative, url_negative):
            ix = min(np.where(score_s < thresholds_opt)[0])
            # thres = thresholds[ix]
            k = k_max_opt - ix
            test_result += bloom_filter_opt.test(url_s, k)
            lookup_negative_logger_dict[k] += 1

        FP_items = test_result + ML_positive
        FPR = FP_items / len(negative_sample)
        print('False positive items: %d' % FP_items)
        print('False positive rate: %f' % FPR)

        lookup_negative_logger_dict[0] = ML_positive
        lookup_negative_logger_dict["FPR"] = FPR
        lookup_negative_logger_dict["size"] = i
        print(f"negative lookup dict: {lookup_negative_logger_dict}")


        mem_arr.append(i)
        FPR_arr.append(FPR)
        num_regions_arr.append(k_max_opt)
        c_arr.append(c_opt)
        region_negatives_arr.append(lookup_negative_logger_dict)

        '''Stage 3: Run PLBF on all the positve samples'''
        ### Test queries
        query_positive = positive_sample.loc[(positive_sample['score'] < thresholds_opt[-2]), 'url']
        score_positive = positive_sample.loc[(positive_sample['score'] < thresholds_opt[-2]), 'score']

        ### False positives from the model:
        ML_positive = len(positive_sample[positive_sample["score"] >=thresholds_opt[-2]])

        lookup_positive_logger_dict = defaultdict(int)

        for score_s, query_s in zip(score_positive, query_positive):
            ix = min(np.where(score_s < thresholds_opt)[0])
            # thres = thresholds[ix]
            k = k_max_opt - ix
            lookup_positive_logger_dict[k] += 1

        lookup_positive_logger_dict[0] = ML_positive
        lookup_positive_logger_dict["FPR"] = FPR
        lookup_positive_logger_dict["size"] = i

        print(f"positive lookup dict: {lookup_positive_logger_dict}")
        region_positives_arr.append(lookup_positive_logger_dict)

        tmp_data = {"memory": mem_arr, "false_positive_rating": FPR_arr}
        tmp_df_data = pd.DataFrame.from_dict(data=tmp_data)
        tmp_df_data.to_csv(f"{args.out_path}tmp_Ada-BF.csv")

        i += args.step
        bar.next()

    print(mem_arr)
    print(FPR_arr)

    data = {"memory": mem_arr, "false_positive_rating": FPR_arr, "num_regions": num_regions_arr, "optimal_c": c_arr}
    df_data = pd.DataFrame.from_dict(data=data)

    df_data.to_csv(f"{args.out_path}Ada-BF.csv")

    df_negative_regions = pd.DataFrame.from_dict(data=region_negatives_arr)
    df_negative_regions.to_csv(f"{args.out_path}/ADA-BF_regions_negatives.csv")

    df_positive_regions = pd.DataFrame.from_dict(data=region_positives_arr)
    df_positive_regions.to_csv(f"{args.out_path}/ADA-BF_regions_positives.csv")
    bar.finish()
