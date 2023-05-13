import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

a = 1.00001
n = 3_000_000

non_keys = np.random.zipf(a, n)
non_keys = non_keys / non_keys.max()

num_keys = 3000
keys = np.random.zipf(a, num_keys)
keys = keys / keys.max()
keys = abs(keys-1)

plt.hist([keys, non_keys], bins=20)   
plt.semilogy()
plt.grid(alpha=0.4)
plt.legend()
plt.title(f'Zipf sample, a={a}, size={n + num_keys}')
plt.savefig('./distributions/img/syntetic_zipfean.png')

df_keys = pd.DataFrame(keys, columns = ['score']).round(15)
df_keys['url'] = df_keys.index
df_keys['label'] = 1

df_non_keys = pd.DataFrame(non_keys, columns = ['score']).round(15)
df_non_keys['url'] = df_non_keys.index + num_keys
df_non_keys['label'] = -1

df = pd.concat([df_keys, df_non_keys], ignore_index=True)

df.to_csv('./data/scores/syntetic_zipfean.csv', index=False)
