#%%
import numpy as np
import pickle
import pandas as pd
#%%
with open('/Users/jayantadey/slide_res_paper/angle/dict-CIFAR100-N2-max250--C3-5x16-bn_F-1024x2000x2000_c20--i2500-lr0.0001-b256--EWC10000.0-1000-s4-64.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
file_to_process = "/Users/jayantadey/slide_res_paper/angle/dict-CIFAR100-N2-max250--C3-5x16-bn_F-1024x2000x2000_c20--i2500-lr0.0001-b256--SI10000.0-0.1"
reps = 10
angles = range(0,184,4)

for angle in angles:
    accuracy = np.zeros(2,dtype=float)

    for rep in range(reps):
        if rep == 0:
            filename = file_to_process + '-' + str(angle) +'.pkl'
        else:
            filename = file_to_process + '-s'+ str(rep) + '-' + str(angle) +'.pkl'

        with open(filename, 'rb') as f:
            data = pickle.load(f)['R']

        accuracy[0] += data['task 1'].iloc[1]
        accuracy[1] += data['task 1'].iloc[2]
        #print(data['task 1'].iloc[1] , data['task 1'].iloc[2])

    accuracy /= reps   
    with open('./reformed_res/si-{}.pickle'.format(angle), 'wb') as f:
        pickle.dump(1-accuracy, f)
# %%
