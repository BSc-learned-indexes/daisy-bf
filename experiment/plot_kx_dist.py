import pandas as pd 
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from datetime import datetime
from progress.bar import Bar
import warnings
warnings.filterwarnings("ignore")



# Arguments 
parser = argparse.ArgumentParser()

parser.add_argument("--file_name", action="store", dest="file_name", type=str, required=True, help="file name to display")
parser.add_argument("--label_name", action="store", dest="label_name", type=str, required=True, help="label name to display")

args = parser.parse_args()

name = args.file_name
label_name = args.label_name

raw_data = pd.read_csv(f'./data/plots/{name}')

data = raw_data.drop(["FPR_target","FPR_actual"], axis=1)
data = data.fillna(0)

print(data.head())
for i, row in data.iterrows():
    FPR_actual = raw_data["FPR_actual"][i]
    label_str = f"{label_name}, FPR: {FPR_actual}"
    plt.clf()
    # row.fillna(0)
    print(len(row))
    x = [*range(len(row))]
    y = row
    xy = zip(x,y)
    new_x = []
    new_y = []
    for (i,j) in xy:
        #if i == 0:
        #    continue
        print(f"i: {i}")
        print(j)
        if j != 0.0:
            new_x.append(i)
            new_y.append(j)

    print(new_x)
    print(new_y)

    # vals = [0]*(len(row)-1)
    # for j in range(1,len(row)):
    #     if j != "nan":
    #         vals[j - 1] = row[j]
    #     print(row[j])
    # plt.hist(range(len(row)), row, log=True, label=label_str)
    # plt.plot(row, label=label_str)
    plt.bar(new_x,new_y, label=label_str)
    plt.xticks(new_x)
    plt.xlabel("Hash functions")
    plt.ylabel("Count")
    plt.title(label_str)
    file_str = f"{label_name}_FPR_{FPR_actual}"
    plt.savefig(f'./distributions/img/daisy_kx_histogram/{file_str}.png')


# # Define plots 
# # Plot distribution of Keys
# x = data["memory"]
# x = x.div(1000)
# y = data["false_positive_rating"]
#
# plt.plot(x, y, linestyle='dashed', marker='x')
#
# #plt.plot(x, y, linestyle='dashed', marker='x')
#
# plt.yscale("log")
# plt.legend(names)
# #plt.axis((50,1550,0.0005,0.5))
# plt.xlabel('Size (Kb)')
# plt.ylabel('False Positive Rate (%)')
#
# now = datetime.now() # current date and time
# date_time = now.strftime("%m-%d-%Y-%H:%M:%S")
# plt.savefig(f'./graphs/fpr_size_{date_time}.png')
#
# bar.finish()
