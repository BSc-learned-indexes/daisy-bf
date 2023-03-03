import pandas as pd 
import argparse
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import warnings
warnings.filterwarnings("ignore")



# Arguments 
parser = argparse.ArgumentParser()

parser.add_argument("--file_names", nargs="+", action="store", dest="file_names", type=str, required=True,
                    help="file names to display")

args = parser.parse_args()

print(args.file_names)

files = args.file_names

bar = Bar('Plotting distributions   ', max=len(files))

for name in files: 
    data = pd.read_csv(f'./data/plots/{name}')

    # Define plots 
    # Plot distribution of Keys
    bar.next()
    x = data["memory"]
    y = data["false_positive_rating"]

    plt.plot(x, y)

plt.yscale("log")
# plt.axis((0,1000,0.0001,0.1))
plt.savefig('./distributions/img/keys.png')

bar.finish()