import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.special import zeta

a = 1.0001
n = 200000
non_keys = np.random.zipf(a, n)

non_keys = non_keys / non_keys.max()

# keys = abs(keys-1) 
keys = abs(non_keys-1) 

# count = np.bincount(keys)
# k = np.arange(1, keys.max() + 1)

# plt.bar(k, count[1:], alpha=0.5, label='sample count')
# plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5, label='expected count')
plt.hist([keys, non_keys], bins=20)   
plt.semilogy()
plt.grid(alpha=0.4)
plt.legend()
plt.title(f'Zipf sample, a={a}, size={n}')
plt.show()


df_keys = pd.DataFrame(keys, columns = ['score']).round(4)
df_keys['url'] = df_keys.index
df_keys['label'] = 1



df_non_keys = pd.DataFrame(non_keys, columns = ['score']).round(4)
df_non_keys['url'] = df_non_keys.index + 200000
df_non_keys['label'] = -1

df = pd.concat([df_keys, df_non_keys], ignore_index=True)

df.to_csv('./data/scores/syntetic_zipfean.csv', index=False)
