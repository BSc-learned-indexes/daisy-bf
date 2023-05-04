import pandas as pd 
import argparse
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
import warnings
warnings.filterwarnings("ignore")

bar = Bar('Plotting distributions   ', max=3)

# Arguments 
parser = argparse.ArgumentParser()

parser.add_argument("--data_set_name", action="store", dest="data_set_name", type=str, required=True,
                    help="set name of the data set")

args = parser.parse_args()

# Path
data = pd.read_csv(f'./data/scores/{args.data_set_name}.csv')


# Define plots 


# Plot distribution of Keys
bar.next()

keys = data.loc[data['label']==1,'score']
non_keys = data.loc[data['label']==-1,'score']


plt.figure(1)
plt.hist(keys, bins=20, log=True, label='Keys')
# x1,x2,y1,y2 = plt.axis()  
plt.axis((0,1,0,1000000))
plt.savefig(f'./distributions/img/{args.data_set_name}_keys.png')

# plt.show()

plt.figure(2)

# plt.style.use('seaborn-deep')


bins = np.linspace(0, 1, 25)

plt.hist([keys, non_keys], bins, log=True, label=['Keys', 'non-Keys'])
plt.legend(loc='upper right')
plt.savefig(f'./distributions/img/{args.data_set_name}_one_plot.png')

# Plot distribution of Non-keys 
bar.next()

plt.figure(3)
plt.hist(non_keys, bins=20, log=True, label='Non-keys')
# x1,x2,y1,y2 = plt.axis()  
plt.axis((0,1,0,1000000))
plt.savefig(f'./distributions/img/{args.data_set_name}_non_keys.png')



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.hist(keys, bins=20, log=True, label='Keys')
ax1.axis((0,1,0,1000000))
ax1.set_title('Keys')
ax1.set_xlabel('Count')
ax1.set_ylabel('Score')


ax2.hist(non_keys, bins=20, log=True, label='Non-keys')
ax2.axis((0,1,0,1000000))
ax2.set_title('Non-keys')
ax2.set_xlabel('Count')
ax2.set_ylabel('Score')




plt.savefig(f'./distributions/img/{args.data_set_name}_sub_plots.png')

bar.next()
bar.finish()