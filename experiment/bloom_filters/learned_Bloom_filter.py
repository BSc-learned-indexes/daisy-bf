import numpy as np
import pandas as pd
import argparse
from Bloom_filter import BloomFilter
import os 


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True,
                    help="path of the dataset")
parser.add_argument('--threshold_min', action="store", dest="min_thres", type=float, required=False, default=0.5,
                    help="Minimum threshold for positive samples")
parser.add_argument('--threshold_max', action="store", dest="max_thres", type=float, required=False, default=0.95,
                    help="Maximum threshold for positive samples")
parser.add_argument('--min_size', action="store", dest="min_size", type=int, default = 100_000, required=False,
                    help="minimum memory to allocate for model + bloom filters")
parser.add_argument('--max_size', action="store", dest="max_size", type=int, default = 500_000, required=False,
                    help="maximum memory to allocate for model + bloom filters")
parser.add_argument('--step', action="store", dest="step", type=int, default = 50_000, required=False,
                    help="amount of memory to step for each")
parser.add_argument('--out_path', action="store", dest="out_path", type=str,
                    required=False, help="path of the output", default="./data/plots/")
parser.add_argument('--model_path', action="store", dest="model_path", type=str, required=True,
                    help="path of the model")




args = parser.parse_args()
DATA_PATH = args.data_path
min_thres = args.min_thres
max_thres = args.max_thres
model_size = os.path.getsize(args.model_path) * 8 #get it to bits instead of byte


# DATA_PATH = './URL_data.csv'
# min_thres = 0.5
# max_thres = 0.9
# size = 200000



'''
Load the data and select training data
'''
data = pd.read_csv(DATA_PATH)
negative_sample = data.loc[(data['label']==-1)]
positive_sample = data.loc[(data['label']==1)]
train_negative = negative_sample.sample(frac = 0.3)


def Find_Optimal_Parameters(max_thres, min_thres, size, train_negative, positive_sample):
    FP_opt = train_negative.shape[0]

    for threshold in np.arange(min_thres, max_thres+10**(-6), 0.01):
        url = positive_sample.loc[(positive_sample['score'] <= threshold),'url']
        n = len(url)
        bloom_filter = BloomFilter(n, size)
        bloom_filter.insert(url)
        ML_positive = train_negative.loc[(train_negative['score'] > threshold),'url']
        bloom_negative = train_negative.loc[(train_negative['score'] <= threshold),'url']
        BF_positive = bloom_filter.test(bloom_negative, single_key=False)
        FP_items = sum(BF_positive) + len(ML_positive)
        print('Threshold: %f, False positive items: %d' %(round(threshold, 2), FP_items))
        if FP_opt > FP_items:
            FP_opt = FP_items
            thres_opt = threshold
            bloom_filter_opt = bloom_filter

    return bloom_filter_opt, thres_opt




'''
Implement learned Bloom filter
'''
if __name__ == '__main__':


    
    mem_arr = []
    FPR_arr = []

    i = args.min_size
    while i <= args.max_size:
        '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
        bloom_filter_opt, thres_opt = Find_Optimal_Parameters(max_thres, min_thres, (i - model_size), train_negative, positive_sample)

        '''Stage 2: Run Ada-BF on all the samples'''
        ### Test URLs
        ML_positive = negative_sample.loc[(negative_sample['score'] > thres_opt), 'url']
        bloom_negative = negative_sample.loc[(negative_sample['score'] <= thres_opt), 'url']
        score_negative = negative_sample.loc[(negative_sample['score'] < thres_opt), 'score']
        BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
        FP_items = sum(BF_positive) + len(ML_positive)
        print('False positive items: %d' % FP_items)

        FPR = FP_items / len(negative_sample)


        mem_arr.append(i)
        FPR_arr.append(FPR)

    data = {"memory": mem_arr, "false_positive_rating": FPR_arr}
    df_data = pd.DataFrame.from_dict(data=data)

    df_data.to_csv(f"{args.out_path}Original_Learned_BF.csv")


