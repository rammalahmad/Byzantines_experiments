import numpy as np
import random

import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import norm as norm_d
from scipy.stats import randint
from scipy.stats import bernoulli
import scipy
from functions import *
from utils import *
from copy import deepcopy
import math
from itertools import permutations
from scipy.spatial.distance import cdist, euclidean




def byz_vr_marina_no_sync(filename, x_init, A, y, clients_A, clients_y, gamma, num_of_byz, p, num_of_workers, attack, agg,
                  sparsificator, sparsificator_params, setting, l2=0, T=50, max_t=np.inf,
                  batch_size=1, save_info_period=100, x_star=None, f_star=None):
    # m -- total number of datasamples
    # n -- dimension of the problem
    m, n = A.shape
    assert (len(x_init) == n)
    assert (len(y) == m)

    #Distributing the data
    G = num_of_workers - num_of_byz
    mul = clients_A[0].shape[0]
    A = A[:mul*G]
    y = y[:mul*G]

    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    

    # it is needed for l-svrg updates
    bernoulli_arr = bernoulli.rvs(p, size=T)
    bernoulli_counter = 0

    G_w = np.zeros((num_of_workers, len(x)))
    G_w1 = logreg_grad(x, [A, y, l2, setting])
    x1 = x.copy()

    for i in range(num_of_workers- num_of_byz):
        G_w[i] = logreg_grad(x, [clients_A[i], clients_y[i], l2, setting])

    # Test on the distribution of data:
    if norm(G_w1 - sum([logreg_grad(x, [clients_A[i], clients_y[i], l2, setting]) for i in range(num_of_workers)]) * 1/num_of_workers) < 1e-10:
        print('Data distributed correctly')
    else:
        print('Problem with Data distribution')

    # Statistics
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    bits_passes = np.array([0.0])
    func_val = np.array([logreg_loss(x, [A, y, l2, setting])- f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    norm_grad = np.array([norm(G_w1)**2])
    t_start = time.time() # Time
    num_of_data_passes = 0.0 # computational complexity
    num_of_bits_passes = 0.0 # communication complexity


    #method
    for it in range(T):
        indices_arr = np.random.choice(mul, size = batch_size, replace=False)
        for i in range(1, num_of_workers):
            indices_arr = np.vstack((indices_arr, np.random.choice(mul, size = batch_size, replace=False)))

        x = x - gamma * G_w1

        # below we emulate the workers behavior and aggregate their updates on-the-fly
        if (bernoulli_arr[bernoulli_counter] == 1):
            num_of_data_passes += 1.0
            num_of_bits_passes += n
        else:
            num_of_data_passes += (2.0 * batch_size / mul)
            num_of_bits_passes += sparsificator_params[0]

        for i in range(num_of_workers - num_of_byz):
            A_i = clients_A[i]
            y_i = clients_y[i]
            if (bernoulli_arr[bernoulli_counter] == 1):
                G_w[i] = logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
            else: 
                grad_diff = logreg_grad(deepcopy(x), [A_i[indices_arr[i]], y_i[indices_arr[i]], l2, setting]) - logreg_grad(x1, [A_i[indices_arr[i]],
                                  y_i[indices_arr[i]], l2, setting])
                
                G_w[i] += sparsificator(grad_diff, sparsificator_params)[0]
            
        #attack
        if attack == "BF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = clients_A[i]
                y_i = clients_y[i]
                G_w[i] = -1 * logreg_grad(deepcopy(x), [A_i, y_i, l2, setting])
            
        if attack == "LF":
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                A_i = clients_A[i]
                y_i = clients_y[i]
                G_w[i] = logreg_grad(deepcopy(x), [A_i, -1 * y_i, l2, setting])

            
        if attack == "IPM":
            sum_of_good = sum(G_w[:num_of_workers - num_of_byz] - G_w1)
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = -0.1 * sum_of_good / (num_of_workers - num_of_byz) + G_w1
                
        if attack == "ALIE":  
            exp_of_good = sum(G_w[:num_of_workers - num_of_byz] - G_w1) / (num_of_workers - num_of_byz)
            var_of_good = (sum((G_w[:num_of_workers - num_of_byz]-G_w1) * (G_w[:num_of_workers - num_of_byz] - G_w1))) / (num_of_workers - num_of_byz) - exp_of_good * exp_of_good
            for i in range(num_of_workers - num_of_byz, num_of_workers):
                G_w[i] = exp_of_good - 1.06 * var_of_good + G_w1

        x1 = deepcopy(x)
        # Aggregation    
        if agg == "GM":
            perm = np.random.permutation(num_of_workers)
            G_w1 = GM(perm, 2, G_w)            
        
        if agg == "M":
            perm = np.random.permutation(num_of_workers)
            G_w1 = np.mean(G_w, axis=0)
        
        if agg == "CM":
            perm = np.random.permutation(num_of_workers)
            G_w1 = CM(perm, 2, G_w)
        
        bernoulli_counter += 1
        
        # Stats save every period of time
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            bits_passes = np.append(bits_passes, num_of_bits_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, setting, 0]) - f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            norm_grad = np.append(norm_grad, [norm(G_w1)**2])
        if tim[-1] > max_t:
            break


    # Stats of the end
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        bits_passes = np.append(bits_passes, num_of_bits_passes)
        func_val = np.append(func_val, logreg_loss(x, [A, y, l2, setting]) - f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        norm_grad = np.append(norm_grad, [norm(G_w1)**2])

    # Save the data
    res = {'last_iter': x, 'func_vals': func_val, 'norm_grad': norm_grad, 'iters': its, 'time': tim, 'data_passes': data_passes, 'bits_passes':bits_passes,
           'squared_distances': sq_distances, 'num_of_workers': num_of_workers, 'num_of_byz': num_of_byz}
    
    with open("dump/" + filename + "_Byz_VR_MARINA_No_Sync_" + attack + "_" + agg + "_gamma_" + str(gamma) + "_l2_" + str(l2) + "_p_" + str(p) + "_epochs_" + str(T) + "_workers_" + str(num_of_workers) + "_batch_" + str(batch_size) + "_byz_" + str(num_of_byz) + ".txt", 'wb') as file:
        pickle.dump(res, file)
    return res