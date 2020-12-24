#%%
import numpy as np
import pickle
import pandas as pd
import os
#%%
with open('/Users/jayantadey/slide_res_paper/data/dict-CIFAR100-N2-max250--C3-5x16-bn_F-1024x2000x2000_c20--i500-lr0.0001-b256--EWC10000.0-1000-0.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
file_to_process1 = "/Users/jayantadey/slide_res_paper/angle/dict-CIFAR100-N2-max250--C3-5x16-bn_F-1024x2000x2000_c20--i500-lr0.0001-b256"
file_to_process2 = "/Users/jayantadey/slide_res_paper/angle/dict-CIFAR100-N2-max250--C3-5x16-bn_F-1024x2000x2000_c20--i2500-lr0.0001-b256"
reps = 10
angles = range(0,184,4)

for angle in angles:
    accuracy = np.zeros(2,dtype=float)

    for rep in range(reps):
        if rep == 0:
            filename = file_to_process1 + '-' + str(angle) +'.pkl'

            if not os.path.exists(filename):
                filename = file_to_process2 + '-' + str(angle) +'.pkl'

        else:
            filename = file_to_process1 + '-s'+ str(rep) + '-' + str(angle) +'.pkl'

            if not os.path.exists(filename):
                filename = file_to_process2 + '-s'+ str(rep) + '-' + str(angle) +'.pkl'

        with open(filename, 'rb') as f:
            data = pickle.load(f)['R']

        accuracy[0] += data['task 1'].iloc[1]
        accuracy[1] += data['task 1'].iloc[2]
        #print(data['task 1'].iloc[1] , data['task 1'].iloc[2])

    accuracy /= reps   
    with open('./reformed_res/none-{}.pickle'.format(angle), 'wb') as f:
        pickle.dump(1-accuracy, f)
# %%
