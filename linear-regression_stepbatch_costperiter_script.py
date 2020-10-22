
import numpy as np

import cvxpy as cp
import sklearn.datasets
from tqdm import tqdm
import itertools
import datetime
from scipy.optimize import fmin_l_bfgs_b
import pickle 
import os
import time


from common_functions import *

##############
date = datetime.date.today()
num_trials = 30
folder = "/scratch/chenggar/distributed-aprox-experiments/"
# folder = "../experiment_data_large_scale/"

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(folder + str(date)):
    os.makedirs(folder + str(date))
##############

##############
prefix="linear-regression_costperiter"
filename = str(date) + "/" + prefix+ "_stepbatch-costperiter-array"+".pickle"
regression = 'linear' # type of regression
source = 'sklearn' # self generate or use sklearn
ran_state = 100
n = 1000
d = 40

def regression_grad(A, b, x):
    return (2 * A.T @ A @ x - 2 * A.T @ b).flatten()
def regression_obj(A, b, x):
    return np.linalg.norm(A @ x - b) ** 2
obj = regression_obj
grad = regression_grad
###############

step_size_lst = np.logspace(-2, 3, 11)
minibatch_size_lst = [1, 4, 8, 16, 32, 64]
#max_sample = 12800
noise_cond_acc_to_max_sample_arr = np.array([[[200,200],[200,200],[200,450],[250,200],[450,200]],
               [[200,200],[200,300],[250,400],[450,400],[700,400]]]) * 64
assert noise_cond_acc_to_max_sample_arr.shape == (2, 5, 2)

method_lst = [method_SGD_solve, method_1_solve_fast, method_2_solve, method_3_solve, method_full_prox_solve_regression] #####
method_name_lst = ["SGD", "method 1" , "method 2", "method 3", "full_prox"]
markers_lst = ['o','s','v','P','*']
colors_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

markers_dict = dict(zip(method_name_lst,markers_lst))
colors_dict = dict(zip(method_name_lst,colors_lst))

noise_lst = [0, 0.5]
condition_lst = [1, 5, 10, 15, 20]
acceleration_lst = [False, True]

dimensions = [len(noise_lst), len(condition_lst), len(acceleration_lst), len(method_lst), len(step_size_lst), len(minibatch_size_lst), num_trials]
if not os.path.exists(folder + filename):
	stepbatch_costperiter_array = np.empty(dimensions, dtype=object)  
	stepbatch_x_array = -1 * np.ones(dimensions + [d], dtype=float)
else:
	with open(folder+filename, 'rb') as handle:
		_noise_cond_acc_to_max_sample_arr, _noise_lst, _condition_lst, _acceleration_lst, _method_name_lst, _num_trials, _step_size_lst, _minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array = pickle.load(handle)

start_time = time.time()
for noise_idx, noise in enumerate(noise_lst):
	for cond_idx, cond in enumerate(condition_lst):
		A,b,opt_cost = generate_data(n,d,source,noise,ran_state,cond,regression,obj)
		for acc_idx, acc in enumerate(acceleration_lst):
			max_sample = noise_cond_acc_to_max_sample_arr[noise_idx, cond_idx, acc_idx]

			pbar_method = tqdm(total=len(method_lst))
			pbar_method.set_description("noise="+str(noise)+"; cond="+ str(cond) + "; acc="+str(acc))
			for method_idx, method in enumerate(method_lst):
				for trial_idx in range(num_trials):
					for step_idx in range(len(step_size_lst)):
						for batch_idx in range(len(minibatch_size_lst)):
							if isinstance(stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], type(np.arange(1))):
								continue
							if not acc:
								stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], x = regression_aprox_method_max_samples(A, b, minibatch_size_lst[batch_idx], step_size_lst[step_idx], method, trial_idx, obj, grad, max_sample = max_sample, verbose = True)
								stepbatch_x_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx] = x
							elif acc:
								stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], x = regression_aprox_method_max_samples_acc(A, b, minibatch_size_lst[batch_idx], step_size_lst[step_idx], method, trial_idx, obj, grad, max_sample = max_sample, verbose = True)
								stepbatch_x_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx] = x
				pbar_method.update(1)
		with open(folder+filename, 'wb') as handle:
			pickle.dump((noise_cond_acc_to_max_sample_arr, noise_lst, condition_lst, acceleration_lst, method_name_lst, num_trials, step_size_lst, minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
print(time.time() - start_time)
							

