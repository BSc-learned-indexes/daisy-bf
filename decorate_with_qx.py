import argparse
import pandas as pd
import math

# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--file_name", action="store", dest="file_name", type=str, required=True, help="File to decorate")
parser.add_argument("--qx_mult", action="store", dest="qx_mult", type=str, required=False, default=100000, help="How many times the highest qx is queried")
args = parser.parse_args()
file_name = args.file_name
qx_mult = args.qx_mult

# Path
df = pd.read_csv(f'./data/scores/{file_name}.csv')

# Decorate with qx
df['qx'] = 1 - df['score']

# Add query count
""""
Elements with a high qx are queried more often.
All elements are queried at least once. The elements with
the highest qx are queried qx_mult times more often than the elements
with the lowest qx.
"""
df['query_count'] = df['qx'].apply(lambda x: math.ceil(x * qx_mult))

# Edge case: if qx is 0, then query_count is 1
df.loc[df['query_count'] == 0.0, 'query_count'] = 1.0
df.to_csv(f'./data/scores/{file_name}_with_qx.csv', index=False)
