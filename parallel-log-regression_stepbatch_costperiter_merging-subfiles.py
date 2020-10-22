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

import multiprocessing as mp

from common_functions import *

##############
date = "2020-10-05"
num_trials = 30
folder = "/scratch/chenggar/distributed-aprox-experiments/"
# folder = "../experiment_data_large_scale/"
date_folder =  str(date) + "/"
subfolder = "parallel_subfolder/"

if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(folder +date_folder):
    os.makedirs(folder + date_folder)
if not os.path.exists(folder + date_folder + subfolder):
    os.makedirs(folder + date_folder + subfolder)
##############

##############
prefix="log-regression_costperiter"

filename = prefix+ "_stepbatch-costperiter-array"+".pickle"
regression = 'log' # type of regression
source = 'sklearn' # self generate or use sklearn
ran_state = 100
n = 1000
d = 40

def log_regression_grad(A, b, x):
    # return needs to be flattened
    assert A.shape[0] == 1
    dummy = A@x
    return (-b/(1 + np.exp(b*dummy)) * A).flatten()
def log_regression_obj(A, b, x):
    return np.sum(np.log(1 + np.exp(-b*(A @ x))))
obj = log_regression_obj
grad = log_regression_grad
###############

step_size_lst = np.logspace(-2, 5, 15)
minibatch_size_lst = [1, 4, 8, 16, 32, 64]
#max_sample = 12800
noise_cond_acc_to_max_sample_arr = np.array([[[250,300],[250,300],[250,400],[400,400],[400,400]],
        [[200,200],[200,200],[200,400],[400,400],[650,400]]]) * 64
assert noise_cond_acc_to_max_sample_arr.shape == (2, 5, 2)

method_lst = [method_SGD_solve, method_1_solve_fast, method_2_solve, method_3_solve, method_full_prox_solve_log_regression] #####
method_name_lst = ["SGD", "method 1" , "method 2", "method 3", "full_prox"]
markers_lst = ['o','s','v','P','*']
colors_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

markers_dict = dict(zip(method_name_lst,markers_lst))
colors_dict = dict(zip(method_name_lst,colors_lst))

noise_lst = [0, 0.01]
condition_lst = [1, 5, 10, 15, 20]
acceleration_lst = [False, True]

dimensions = [len(noise_lst), len(condition_lst), len(acceleration_lst), len(method_lst), len(step_size_lst), len(minibatch_size_lst), num_trials]
if not os.path.exists(folder +date_folder+ filename):
	stepbatch_costperiter_array = np.empty(dimensions, dtype=object)  
	stepbatch_x_array = -1 * np.ones(dimensions + [d], dtype=float)
else:
	with open(folder+date_folder+filename, 'rb') as handle:
		_noise_cond_acc_to_max_sample_arr, _noise_lst, _condition_lst, _acceleration_lst, _method_name_lst, _num_trials, _step_size_lst, _minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array = pickle.load(handle)

param_lst = itertools.product(list(enumerate(noise_lst)), list(enumerate(condition_lst)), list(enumerate(acceleration_lst)))

for noise_idx, noise in enumerate(noise_lst):
	for cond_idx, cond in enumerate(condition_lst):
		for acc_idx, acc in enumerate(acceleration_lst):
			identifier = "_".join(str(noise_idx) + str(cond_idx) + str(acc_idx))
			subfile_name =  prefix+ "_stepbatch-costperiter-array_" + identifier + ".pickle"
			with open(folder+date_folder + subfolder + subfile_name, 'rb') as handle:
				noise_tup, condition_tup, acceleration_tup, method_name_lst, num_trials, step_size_lst, minibatch_size_lst, sub_stepbatch_costperiter_array, sub_stepbatch_x_array = pickle.load(handle)
			stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx] = sub_stepbatch_costperiter_array
			stepbatch_x_array[noise_idx, cond_idx, acc_idx] = sub_stepbatch_x_array
with open(folder+date_folder+filename, 'wb') as handle:
	pickle.dump((noise_cond_acc_to_max_sample_arr, noise_lst, condition_lst, acceleration_lst, method_name_lst, num_trials, step_size_lst, minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array), handle, protocol=pickle.HIGHEST_PROTOCOL)






