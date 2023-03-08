import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
from random import randint
import argparse
import os
from progress.bar import Bar
import math



def hashfunc(m, hash_seed=None):
    ss = randint(1, 99999999)
    def hash_m(x):
        if hash_seed is None:
            return murmurhash3_32(x,seed=ss)%m
        else:
            #print(f"hash_seed: {hash_seed}")
            return murmurhash3_32(x,seed=hash_seed)%m
    return hash_m


'''
Class for Standard Bloom filter
'''
class BloomFilter():
    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = int(hash_len)
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1,int(self.hash_len/n*0.6931472))
        elif (self.n==0):
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(hashfunc(self.hash_len, hash_seed=i))
        self.table = np.zeros(self.hash_len, dtype=int)
    def insert(self, key):
        if self.hash_len == 0:
            raise SyntaxError('cannot insert to an empty hash table')
        for i in key:
            for j in range(self.k):
                t = self.h[j](i)
                self.table[t] = 1

    def test(self, keys, single_key = True):
        if single_key:
            test_result = 0
            match = 0
            if self.hash_len > 0:
                for j in range(self.k):
                    t = self.h[j](keys)
                    match += 1 * (self.table[t] == 1)
                if match == self.k:
                    test_result = 1
        else:
            test_result = np.zeros(len(keys))
            ss=0
            if self.hash_len > 0:
                for key in keys:
                    match = 0
                    for j in range(self.k):
                        t = self.h[j](key)
                        match += 1*(self.table[t] == 1)
                    if match == self.k:
                        test_result[ss] = 1
                    ss += 1
        return test_result

'''Run Bloom filter'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                        help="path of the dataset")
    parser.add_argument('--min_size', action="store", dest="min_size", type=int, default = 150_000,
                        help="minimum memory to allocate for model + bloom filters")
    parser.add_argument('--max_size', action="store", dest="max_size", type=int, default = 500_000,
                        help="maximum memory to allocate for model + bloom filters")
    parser.add_argument('--step', action="store", dest="step", type=int, default = 50_000,
                        help="amount of memory to step for each")
    parser.add_argument('--out_path', action="store", dest="out_path", type=str,
                        required=False, help="path of the output", default="./data/plots/")
    args = parser.parse_args()


    MEM_ARR = []
    FPR_ARR = []



    DATA_PATH   = args.data_path
    MAX_SIZE    = args.max_size
    MIN_SIZE    = args.min_size
    STEP        = args.step
    OUT_PATH    = args.out_path


    #Progress bar 
    bar = Bar('Creating standard BF     ', max=math.floor((MAX_SIZE - MIN_SIZE)/STEP))

    data = pd.read_csv(DATA_PATH)
    negative_sample = data.loc[(data['label'] == -1)]
    positive_sample = data.loc[(data['label'] == 1)]
    url = positive_sample['url']
    n = len(url)

    # Test all size variations
    i = MIN_SIZE
    while i <= MAX_SIZE:
        bar.next()
        print(f"current memory allocated: {i}")
        bloom_filter = BloomFilter(n, i)
        bloom_filter.insert(url)
        url_negative = negative_sample['url']
        n1 = bloom_filter.test(url_negative, single_key=False)
        FPR = sum(n1) / len(negative_sample)
        MEM_ARR.append(i)
        FPR_ARR.append(FPR)
        print('False positive rate: ', FPR)
        i += STEP

    
    # Write results to csv

    data = {"memory": MEM_ARR, "false_positive_rating": FPR_ARR}
    df_data = pd.DataFrame.from_dict(data=data)

    df_data.to_csv(f"{OUT_PATH}Standard_BF.csv")

    bar.finish()
