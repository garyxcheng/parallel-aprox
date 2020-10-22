import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
# from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt

import cvxpy as cp
import sklearn.datasets
from tqdm import tqdm
import itertools
import datetime
from scipy.optimize import fmin_l_bfgs_b
import pickle 
import os

from common_functions import *

#####################
#HELPER functions
#####################
def make_plot1(max_sample, method_name_lst, minibatch_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, folder, prefix):
    #PLOT 1 single plot showing speedup vs minibatch for best step size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_idx, method_name in enumerate(method_name_lst):
        stepbatch_array = np.minimum(methodstepbatchtrial_arr[method_idx], max_sample)

        assert minibatch_size_lst[0] == 1
        minibatch_1 = stepbatch_array[:, 0, :]
        fastest_minibatch_1 = np.min(minibatch_1, axis=0)
        
        assert fastest_minibatch_1.shape == (methodstepbatchtrial_arr.shape[-1],)
        
        #numpy intelligently matching dimensions, so we need to make sure it is doing it when we want it to
        assert stepbatch_array.shape[2] == fastest_minibatch_1.shape[0] and stepbatch_array.shape[2] != stepbatch_array.shape[1] and stepbatch_array.shape[2] != stepbatch_array.shape[0]
        
        fastest_stepbatch_array = np.min(stepbatch_array, axis = 0)
        assert fastest_stepbatch_array.shape == (len(minibatch_size_lst), methodstepbatchtrial_arr.shape[-1])
        
        speedup_ratio = fastest_minibatch_1 / fastest_stepbatch_array
        assert speedup_ratio.shape == (len(minibatch_size_lst), methodstepbatchtrial_arr.shape[-1])

        #compute mean output
        mean_output_lst = np.median(speedup_ratio, axis= 1)

        #compute 95 percentile output
        ninefive_output_lst = np.percentile(speedup_ratio, 95, axis= 1)

        #compute 5 percentile output
        five_output_lst = np.percentile(speedup_ratio, 5, axis= 1)

        ax.plot(minibatch_size_lst, mean_output_lst, label=method_name, marker=markers_lst[method_idx], color= colors_lst[method_idx])
        ax.fill_between(minibatch_size_lst, five_output_lst, ninefive_output_lst, color=colors_lst[method_idx], alpha=0.1)
    # ax.set_title("Speed up for best step sizes")
    # ax.set_ylabel("speedup compared to minibatch = 1 for eps = " + str(eps))
    # ax.set_xlabel("minibatch size")
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.legend( prop={'size': 10})
    fig.savefig(folder + prefix + "plot1.pdf")
    plt.close()

def make_plot2(batch_idx, max_sample, method_name_lst, step_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, folder, prefix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_idx, method_name in enumerate(method_name_lst):
   
        stepbatch_array = np.minimum(methodstepbatchtrial_arr[method_idx], max_sample)
        
        # assert minibatch_size_lst[0] == 1
        minibatch_1 = stepbatch_array[:, 0, :]
        fastest_minibatch_1 = np.min(minibatch_1, axis=0)
        
        #numpy intelligently matching dimensions, so we need to make sure it is doing it when we want it to
        assert stepbatch_array.shape[2] == fastest_minibatch_1.shape[0] and stepbatch_array.shape[2] != stepbatch_array.shape[1] and stepbatch_array.shape[2] != stepbatch_array.shape[0]
        speedup_ratio = fastest_minibatch_1 / stepbatch_array
        
        
        #compute mean output
        mean_arr = np.median(speedup_ratio, axis= 2)
        mean_output_lst = mean_arr[:, batch_idx]
        
        #compute 95 percentile output
        ninefive_arr =  np.percentile(speedup_ratio, 95, axis= 2)
        ninefive_output_lst = ninefive_arr[:, batch_idx]

        #compute 5 percentile output
        five_arr =  np.percentile(speedup_ratio, 5, axis= 2)
        five_output_lst = five_arr[:, batch_idx]
    
        ax.plot(step_size_lst, mean_output_lst, label=method_name, marker=markers_lst[method_idx], color= colors_lst[method_idx])
        ax.fill_between(step_size_lst, five_output_lst, ninefive_output_lst, color=colors_lst[method_idx], alpha=0.1)
#     ax.set_title(prefix + " with minibatch size = " + str(minibatch))
#     ax.set_ylabel("speedup compared to minibatch = 1 for eps = " + str(eps))
#     ax.set_xlabel("step size")
    ax.set_xscale('log')
    ax.legend( prop={'size': 10})
    fig.savefig(folder + prefix + "plot2_minibatch=" + str(minibatch)+ ".pdf")
    plt.close()

def make_plot3(batch_idx, max_sample, method_name_lst, step_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, folder, prefix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_idx, method_name in enumerate(method_name_lst):
        #compute mean output
        mean_arr = np.median(np.minimum(methodstepbatchtrial_arr[method_idx], max_sample), axis= 2)
        mean_output_lst = mean_arr[:, batch_idx]
        
        #compute 95 percentile output
        ninefive_arr =  np.percentile(np.minimum(methodstepbatchtrial_arr[method_idx], max_sample), 95, axis= 2)
        ninefive_output_lst = ninefive_arr[:, batch_idx]

        #compute 5 percentile output
        five_arr =  np.percentile(np.minimum(methodstepbatchtrial_arr[method_idx], max_sample), 5, axis= 2)
        five_output_lst = five_arr[:, batch_idx]
        
        ax.plot(step_size_lst, mean_output_lst, label=method_name, marker=markers_lst[method_idx], color= colors_lst[method_idx])
        ax.fill_between(step_size_lst, five_output_lst, ninefive_output_lst, color=colors_lst[method_idx], alpha=0.1)
#     ax.set_title(prefix + " with minibatch size = " + str(minibatch))
#     ax.set_ylabel("iterations to get to error eps = " + str(eps))
#     ax.set_xlabel("step size")
    ax.set_xscale('log')
    ax.legend(prop={'size': 10})
    fig.savefig(folder + prefix + "plot3_minibatch=" + str(minibatch)+ ".pdf")
    plt.close()
####################

##############
date = "2020-09-23"
folder = "/scratch/chenggar/distributed-aprox-experiments/"
if not os.path.exists(folder):
    raise Exception()
if not os.path.exists(folder + str(date)):
    raise Exception() 

plt_date = datetime.date.today()
plt_folder="plots_large_scale/abs_reg/"
if not os.path.exists(plt_folder):
    os.makedirs(plt_folder)
if not os.path.exists(plt_folder + str(plt_date)):
    os.makedirs(plt_folder + str(plt_date))
##############

##############
prefix="abs-regression_costperiter"
filename = str(date) + "/" + prefix+ "_stepbatch-costperiter-array"+".pickle"
regression = 'abs' # type of regression
source = 'sklearn' # self generate or use sklearn
ran_state = 100
n = 1000
d = 40

def l1_regression_grad(A, b, x):
    # return needs to be flattened
    assert A.shape[0] == 1
    return (np.sign(A@x -b) * A).flatten()
def l1_regression_obj(A, b, x):
    return np.linalg.norm(A @ x - b, ord=1)
obj = l1_regression_obj
grad = l1_regression_grad

OLD_method_name_lst = ["SGD", "method 1" , "method 2", "method 3", "full_prox"] # TODO
method_name_lst = ["SGM", "AvTrunc" , "TruncAv", "IA", "Prox"] # TODO
markers_lst = ['o','s','v','P','*']
colors_lst = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

markers_dict = dict(zip(method_name_lst,markers_lst))
colors_dict = dict(zip(method_name_lst,colors_lst))

trunc_val = None
eps = 0.05
###############

#Plots 1,2,3
with open(folder+filename, 'rb') as handle:
    noise_cond_acc_to_max_sample_arr, noise_lst, condition_lst, acceleration_lst, _method_name_lst, num_trials, step_size_lst, minibatch_size_lst, stepbatch_costperiter_array, stepbatch_x_array = pickle.load(handle)
    assert np.all([_method_name_lst[i] == OLD_method_name_lst[i] for i in range(len(_method_name_lst))])
    
eps_stebatch_arr = -1 * np.ones(stepbatch_costperiter_array.shape)
allowinf_eps_stebatch_arr = -1 * np.ones(stepbatch_costperiter_array.shape)
for noise_idx, noise in enumerate(noise_lst):
    for cond_idx, cond in enumerate(condition_lst):
        A,b,opt_cost = generate_data(n,d,source,noise,ran_state,cond,regression,obj)
        eps_stebatch_arr[noise_idx, cond_idx] = vectorized_first_idx_below_eps(eps, opt_cost, stepbatch_costperiter_array[noise_idx, cond_idx], trunc_val = None)
        allowinf_eps_stebatch_arr[noise_idx, cond_idx] = vectorized_first_idx_below_eps(eps, opt_cost, stepbatch_costperiter_array[noise_idx, cond_idx], trunc_val = np.inf)

for noise_idx, noise in enumerate(noise_lst):
    for cond_idx, cond in enumerate(condition_lst):
        for acc_idx, acc in enumerate(acceleration_lst):
            methodstepbatchtrial_arr = eps_stebatch_arr[noise_idx, cond_idx, acc_idx]
            max_sample = noise_cond_acc_to_max_sample_arr[noise_idx, cond_idx, acc_idx]

            pbar_method = tqdm(total=len(minibatch_size_lst))
            pbar_method.set_description("noise="+str(noise)+"; cond="+ str(cond) + "; acc="+str(acc))

            make_plot1(max_sample, method_name_lst, minibatch_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, plt_folder + str(plt_date) +"/", prefix+"noise="+str(noise)+"-cond="+ str(cond) + "-acc="+str(acc)+"_")
            for batch_idx, minibatch in enumerate(minibatch_size_lst):
                make_plot2(batch_idx, max_sample, method_name_lst, step_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, plt_folder + str(plt_date) +"/", prefix+"noise="+str(noise)+"-cond="+ str(cond) + "-acc="+str(acc)+"_")
                make_plot3(batch_idx, max_sample, method_name_lst, step_size_lst, methodstepbatchtrial_arr, markers_lst, colors_lst, plt_folder + str(plt_date) +"/", prefix+"noise="+str(noise)+"-cond="+ str(cond) + "-acc="+str(acc)+"_")
                pbar_method.update(1)
            pbar_method.close()
#TODO maybe split up performance plotting from the rest
#Performance plots
# [len(noise_lst), len(condition_lst), len(acceleration_lst), len(method_lst), len(step_size_lst), len(minibatch_size_lst), num_trials]
flat_method_to_alltrials = np.transpose(eps_stebatch_arr, [2, 3, 0, 1, 4, 5, 6]).reshape((2, 5, -1))
allowinf_flat_method_to_alltrials = np.transpose(allowinf_eps_stebatch_arr, [2, 3, 0, 1, 4, 5, 6]).reshape((2, 5, -1))
best_arr = np.min(flat_method_to_alltrials, axis = 1, keepdims=False)
# allowinf_best_arr = np.min(allowinf_flat_method_to_alltrials, axis = 1, keepdims=True)
num_inf_arr = np.count_nonzero(allowinf_flat_method_to_alltrials == np.inf, axis = 1, keepdims=False)

max_ratio_cutoff = 500
keep_trial_lim = 2
for acc_idx, acc in enumerate(acceleration_lst):
    num_inf_subarr = num_inf_arr[acc_idx]
    flatten_ratio_arr = np.minimum(np.nan_to_num(flat_method_to_alltrials[acc_idx][:, num_inf_subarr <= keep_trial_lim] / best_arr[acc_idx][num_inf_subarr<= keep_trial_lim], nan = 1, posinf = np.inf), max_ratio_cutoff)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for method_idx, method_name in enumerate(method_name_lst):
        values, base = np.histogram(flatten_ratio_arr[method_idx], bins=50)
        #evaluate the cumulative
        cumulative = np.cumsum(values) / np.size(flatten_ratio_arr[method_idx])
        ax.plot(base[:-1], cumulative, label = method_name, color = colors_lst[method_idx])
        ax.hlines(1, base[-2], max_ratio_cutoff, color=colors_lst[method_idx])
    ax.legend()
    ax.set_xscale('log')
    fig.savefig(plt_folder + str(plt_date) +"/"  + prefix + "_acc=" + str(acc)+"_perfplot.pdf")
    plt.close()
# plt.xlim([1, 1000])
# plt.title("performance plot (linear eps= 0.05; allow ratio equal infinity; no dropping)")
# plt.ylabel("fraction of trials which satisfy performance threshold")
# plt.xlabel("performance threshold - ratio of time to eps for algorithm against best algorithm for trial")








