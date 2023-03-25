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

parser.add_argument("--file_names", nargs="+", action="store", dest="file_names", type=str, required=True,
                    help="file names to display")

args = parser.parse_args()

names = args.file_names

bar = Bar('Plotting distributions   ', max=len(names))

for name in names: 
    data = pd.read_csv(f'./data/plots/{name}')

    # Define plots 
    # Plot distribution of Keys
    bar.next()
    x = data["memory"]
    x = x.div(1000)
    y = data["false_positive_rating"]

    plt.plot(x, y, linestyle='dashed', marker='x')

#plt.plot(x, y, linestyle='dashed', marker='x')

plt.yscale("log")
plt.legend(names)
#plt.axis((50,1550,0.0005,0.5))
plt.xlabel('Size (Kb)')
plt.ylabel('False Positive Rate (%)')

now = datetime.now() # current date and time
date_time = now.strftime("%m-%d-%Y-%H:%M:%S")
plt.savefig(f'./graphs/fpr_size_{date_time}.png')

bar.finish()