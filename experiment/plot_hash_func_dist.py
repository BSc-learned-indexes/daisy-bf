
import pandas as pd 
import argparse
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import warnings
from os import listdir
from os.path import isfile, join

warnings.filterwarnings("ignore")

# Arguments 
parser = argparse.ArgumentParser()

# parser.add_argument("--train_split", action="store", dest="train_split", type=float, required=False,
#                     help="split of training data", default = 0.3)

args = parser.parse_args()

path = "./data/plots/daisy_out/"

log_y_axis=False

files = [f for f in listdir(path) if isfile(join(path, f))]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
keys = []
non_keys = []
for file in files:
    data = pd.read_csv(path + file)
    keys.append(data.loc[data['label']==1,'kx'])
    non_keys.append(data.loc[data['label']==-1,'kx'])

    
ax1.hist(keys, log=log_y_axis, label='Keys')
#ax1.axis((0,1,0,1000000))
ax1.set_title('Keys')
ax1.set_ylabel('Count')
ax1.set_xlabel('Number of hash functions')


ax2.hist(non_keys, log=log_y_axis, label='Non-keys')
#ax2.axis((0,1,0,1000000))
ax2.set_title('Non-keys')
ax2.set_ylabel('Count')
ax2.set_xlabel('Number of hash functions')




plt.savefig('./distributions/img/daisy_out/subplots.png')