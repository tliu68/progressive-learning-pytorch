#%%
import numpy as np
import pickle
#%%
with open('store/results/dict-CIFAR100-N10--C3-5x16-bn_F-1024x2000x2000_c100--i5000-lr0.0001-b256--offline-1-1.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
