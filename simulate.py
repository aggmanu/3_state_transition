"""
functions for generating binary variables
"""
import numpy as np
import function as ft
import matplotlib.pyplot as plt

from numba import jit

#=========================================================================================
"""generate binary time-series 
    input: interaction matrix w[n,n], interaction variance g, data length l
    output: time series s[l,n]
""" 
@jit(nopython=True)
def generate_data(w,l,n, initial=None):
    #n = np.shape(w)[0]
    s = np.ones((l,n))
    #all_p = np.zeros((l-1, n))
    #all_p_rnd = np.zeros((l-1, n))

    s[0, :] = initial

    rand_nums = np.random.rand(l-1, n)
    
    for t in range(l-1):
        #print('Simulating time ', t, ' out of ', l, end='\r')
        #h = np.sum(w[:,:]*s[t,:],axis=1) # Wij from j to i

        #h = np.matmul(w, s[t, :])
        #p = 1/(1+np.exp(-2*h))
        #s[t+1,:]= ft.sign_vec(p-rand_nums[t, :])

        for i in range(n):
            h = np.dot(w[i, :], s[t, :])
            p = 1/(1+np.exp(-2*h))
            if rand_nums[t, i] < p:
                #print(p, rand_nums[t, i])
                s[t+1, i] = -1

        #p = np.tanh(h)
        #all_p[t, :] = p
        #all_p_rnd[t, :] = p - np.random.rand(n)
        #if t>1:
        #    diff_p = np.amax(np.abs(p - p_prev))
        #    if diff_p > 0:
        #        print(diff_p, 'time', t)
        #        input('w')
        #plt.plot(h)


        #s[t+1,:]= ft.sign_vec(p-0.5)
        #p_prev = p
        #s[t+1,:]= ft.sign_vec(np.tanh(h))

    #flat_p = all_p.flatten()
    #plt.hist(all_p.flatten())
    #plt.show()
    #input('w')
    #plt.hist(all_p_rnd.flatten())
    #plt.show()
    #input('w')

    #max_p = np.amax(all_p)
    #min_p = np.amin(all_p)
    #diff_p = max_p -  min_p
    ##all_p = -3 + 6*(all_p - min_p)/diff_p
    ##all_exp = np.tanh(all_p)
    ##print(np.amax(all_exp), np.amin(all_exp))
    #pos = np.where(all_p > 0)
    #neg = np.where(all_p < 0)
    #all_p[pos] = 1
    #all_p[neg] = -1
    ##s[t+1,:]= ft.sign_vec(p-np.random.rand(n))
    ##plt.show()
    #return all_p
    #print(np.where(s == -1))
    #plt.imshow(s)
    #plt.show()
    return s

