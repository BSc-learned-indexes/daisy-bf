import pandas as pd 
import argparse
import matplotlib.pyplot as plt
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
    x = [*range(len(row))]
    y = row
    xy = zip(x,y)
    new_x = []
    new_y = []
    for (i,j) in xy:
        if j != 0.0:
            new_x.append(i)
            new_y.append(j)

    plt.bar(new_x,new_y, label=label_str)
    plt.xticks(new_x)
    plt.xlabel("Hash functions")
    plt.ylabel("Count")
    plt.title(label_str)
    file_str = f"{label_name}_FPR_{FPR_actual}"
    plt.savefig(f'./distributions/img/daisy_kx_histogram/{file_str}.png')
