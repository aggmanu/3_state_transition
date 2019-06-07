import numpy as np
import time

import matplotlib.pyplot as plt
import inference as infer
import simulate

#data = np.load('/Users/aggarwalm4/Downloads/codesource/3_state_data.npy')
#
#data = data[:, 1]
#
#N = len(data)
#
#data_transitions = np.zeros((N-1, 9))
#
#for i in range(1, N):
#    print(i, end='\r')
#    if data[i] == data[i-1]:
#        if data[i] == 1:
#            data_transitions[i-1, 0] = 1
#        elif data[i] == 2:
#            data_transitions[i-1, 2] = 1
#        elif data[i] == 3:
#            data_transitions[i-1, 4] = 1
#        else:
#            print('unexpected')
#            print(data[i-1], data[i])
#            input('w')
#    else:
#        if data[i-1] == 1 and data[i] == 2:
#            data_transitions[i-1, 1] = 1
#        elif data[i-1] == 2 and data[i] == 3:
#            data_transitions[i-1, 3] = 1
#        elif data[i-1] == 3 and data[i] == 2:
#            data_transitions[i-1, 5] = 1
#        elif data[i-1] == 2 and data[i] == 1:
#            data_transitions[i-1, 6] = 1
#        elif data[i-1] == 1 and data[i] == 3:
#            data_transitions[i-1, 7] = 1
#        elif data[i-1] == 3 and data[i] == 1:
#            data_transitions[i-1, 8] = 1
#        else:
#            print('unexpected')
#            print(data[i-1], data[i])
#            input('w')
#
#sums = np.sum(data_transitions, axis=0)
#tot = np.sum(sums)
#prob = sums/tot
#print(prob)
#np.save('/Users/aggarwalm4/Downloads/data_transitions', data_transitions)






data = np.load('/Users/aggarwalm4/Downloads/data_transitions.npy')
N, dummy = data.shape

#n_tran = (N + np.sum(data, axis = 0))/2
#print(n_tran)


w = infer.fem(data)

L = 1000000
n = 9

initial = data[0,:]

start = time.time()
data_sim = simulate.generate_data(w, L, n, initial)
finish = time.time()
print('\n time taken to simulate', finish - start)

not_same = np.where(data[:L, :].flatten() - data_sim.flatten() != 0)[0]
print('\n the differences are at')
print(not_same, len(not_same))




##print(w)
##plt.imshow(w)
##plt.show()
#
##print(data)
#
