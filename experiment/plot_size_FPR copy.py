import pandas as pd 
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
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
    #x = x.div(1000)
    y = data["false_positive_rating"]

    plt.plot(x, y, linestyle='dashed', marker='x')

x = [4205335, 3305476, 3033411, 2879589, 2732328, 2631061, 2581260, 2485739, 2431056, 2401464, 2331858]
x = x / 1000

y = [5e-05, 0.000395, 0.00074, 0.001085, 0.00143, 0.001775, 0.00212, 0.0024649999999999997, 0.0028099999999999996, 0.0031549999999999994, 0.003499999999999999]
plt.plot(x, y, linestyle='dashed', marker='x')

# plt.yscale("log")
plt.legend(names)
plt.axis((50,1550,0.0005,0.5))
plt.xlabel('Size (Kb)')
plt.ylabel('False Positive Rate (%)')
plt.savefig('./graphs/final.png')

bar.finish()