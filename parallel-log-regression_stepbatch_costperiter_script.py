
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
date = datetime.date.today()
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

# param_lst = itertools.product(list(enumerate(noise_lst)), list(enumerate(condition_lst)), list(enumerate(acceleration_lst)), list(enumerate(method_lst)), list(enumerate(np.arange(num_trials))), list(enumerate(step_size_lst)), list(enumerate(minibatch_size_lst)))
# def run_experiment(input_val):
# 	noise_tup, condition_tup, acceleration_tup, method_tup, trail_tup, step_size_tup, minibatch_tup = input_val[0], input_val[1], input_val[2], input_val[3], input_val[4], input_val[5], input_val[6]
# 	A,b,opt_cost = generate_data(n,d,source,noise_tup[1],ran_state,condition_tup[1],regression,obj)
# 	max_sample = noise_cond_acc_to_max_sample_arr[noise_tup[0], condition_tup[0], acceleration_tup[0]]
# 	# if isinstance(stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], type(np.arange(1))):
# 	# 	continue
# 	if not acceleration_tup[1]:
# 		stepbatch_costperiter_array[noise_tup[0], condition_tup[0], acceleration_tup[0], method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]], x = regression_aprox_method_max_samples(A, b, minibatch_tup[1], step_size_tup[1], method_tup[1], trail_tup[1], obj, grad, max_sample = max_sample, verbose = True)
# 		stepbatch_x_array[noise_tup[0], condition_tup[0], acceleration_tup[0], method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]] = x
# 	elif acceleration_tup[1]:
# 		stepbatch_costperiter_array[noise_tup[0], condition_tup[0], acceleration_tup[0], method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]], x = regression_aprox_method_max_samples_acc(A, b, minibatch_tup[1], step_size_tup[1], method_tup[1], trail_tup[1], obj, grad, max_sample = max_sample, verbose = True)
# 		stepbatch_x_array[noise_tup[0], condition_tup[0], acceleration_tup[0], method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]] = x



param_lst = itertools.product(list(enumerate(noise_lst)), list(enumerate(condition_lst)), list(enumerate(acceleration_lst)))
#  list(enumerate(method_name_lst)), list(enumerate(np.arange(num_trials))), list(enumerate(step_size_lst)), list(enumerate(minibatch_size_lst))
def run_experiment(input_val):
	noise_tup, condition_tup, acceleration_tup= input_val[0], input_val[1], input_val[2]
	identifier = "_".join(str(noise_tup[0]) + str(condition_tup[0]) + str(acceleration_tup[0]))
	# , trail_tup, step_size_tup, minibatch_tup 

	dimensions = [len(method_lst), len(step_size_lst), len(minibatch_size_lst), num_trials]
	sub_stepbatch_costperiter_array = np.empty(dimensions, dtype=object)  
	sub_stepbatch_x_array = -1 * np.ones(dimensions + [d], dtype=float)

	A,b,opt_cost = generate_data(n,d,source,noise_tup[1],ran_state,condition_tup[1],regression,obj)
	max_sample = noise_cond_acc_to_max_sample_arr[noise_tup[0], condition_tup[0], acceleration_tup[0]]
	for method_tup in enumerate(method_lst):
		for trail_tup in enumerate(range(num_trials)):
			for step_size_tup in enumerate(step_size_lst):
				for minibatch_tup in enumerate(minibatch_size_lst):
					# if isinstance(stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], type(np.arange(1))):
					# 	continue
					if not acceleration_tup[1]:
						sub_stepbatch_costperiter_array[method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]], x = regression_aprox_method_max_samples(A, b, minibatch_tup[1], step_size_tup[1], method_tup[1], trail_tup[1], obj, grad, max_sample = max_sample, verbose = True)
						sub_stepbatch_x_array[method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]] = x
					elif acceleration_tup[1]:
						sub_stepbatch_costperiter_array[method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]], x = regression_aprox_method_max_samples_acc(A, b, minibatch_tup[1], step_size_tup[1], method_tup[1], trail_tup[1], obj, grad, max_sample = max_sample, verbose = True)
						sub_stepbatch_x_array[method_tup[0], step_size_tup[0], minibatch_tup[0], trail_tup[0]] = x
				
	subfile_name =  prefix+ "_stepbatch-costperiter-array_" + identifier + ".pickle"
	with open(folder+date_folder + subfolder + subfile_name, 'wb') as handle:
		pickle.dump((noise_tup, condition_tup, acceleration_tup, method_name_lst, num_trials, step_size_lst, minibatch_size_lst, sub_stepbatch_costperiter_array, sub_stepbatch_x_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("###########")
	print("DONE")
	print(noise_tup)
	print(condition_tup)
	print(acceleration_tup)
	print("###########")
p = mp.Pool(20)
p.map(run_experiment, param_lst)
p.close()
p.join()




# with open(folder+filename, 'wb') as handle:
# 	pickle.dump((noise_cond_acc_to_max_sample_arr, noise_lst, condition_lst, acceleration_lst, method_name_lst, num_trials, step_size_lst, minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(time.time() - start_time)




# for noise_idx, noise in enumerate(noise_lst):
# 	for cond_idx, cond in enumerate(condition_lst):
# 		A,b,opt_cost = generate_data(n,d,source,noise,ran_state,cond,regression,obj)
# 		for acc_idx, acc in enumerate(acceleration_lst):
# 			max_sample = noise_cond_acc_to_max_sample_arr[noise_idx, cond_idx, acc_idx]

# 			pbar_method = tqdm(total=len(method_lst))
# 			pbar_method.set_description("noise="+str(noise)+"; cond="+ str(cond) + "; acc="+str(acc))
# 			for method_idx, method in enumerate(method_lst):
# 				for trial_idx in range(num_trials):
# 					for step_idx in range(len(step_size_lst)):
# 						for batch_idx in range(len(minibatch_size_lst)):
# 							if isinstance(stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], type(np.arange(1))):
# 								continue
# 							if not acc:
# 								stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], x = regression_aprox_method_max_samples(A, b, minibatch_size_lst[batch_idx], step_size_lst[step_idx], method, trial_idx, obj, grad, max_sample = max_sample, verbose = True)
# 								stepbatch_x_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx] = x
# 							elif acc:
# 								stepbatch_costperiter_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx], x = regression_aprox_method_max_samples_acc(A, b, minibatch_size_lst[batch_idx], step_size_lst[step_idx], method, trial_idx, obj, grad, max_sample = max_sample, verbose = True)
# 								stepbatch_x_array[noise_idx, cond_idx, acc_idx, method_idx, step_idx, batch_idx, trial_idx] = x
# 				pbar_method.update(1)
# 			pbar_method.close()
							

