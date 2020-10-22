import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import cvxpy as cp
import sklearn.datasets
from tqdm import tqdm_notebook
import itertools
import datetime
from scipy.optimize import fmin_l_bfgs_b

def first_idx_below_eps(eps, opt_cost, cost_lst, trunc_val = None, soft_require=False):
    if not soft_require:
        assert np.all(opt_cost <= cost_lst[~np.isnan(cost_lst)]), np.min(cost_lst - opt_cost)
    else:
        assert np.all(opt_cost- 1e-6 <= cost_lst[~np.isnan(cost_lst)]), np.min(cost_lst - opt_cost)
    idx_lst = np.argwhere(cost_lst - opt_cost <= eps)
    if idx_lst.size == 0:
        if trunc_val is None:
            return cost_lst.size
        return trunc_val
    return np.float(np.min(idx_lst)) + 1
vectorized_first_idx_below_eps = np.vectorize(first_idx_below_eps)

def make_eps_stepbatch_arr(eps, opt_cost, method_to_stepbatch_costperiter_array, trunc_val = None):
    method_to_stepbatch_array = {method_name:-1 * np.ones(method_to_stepbatch_costperiter_array[method_name].shape, dtype=np.float) for method_name in method_to_stepbatch_costperiter_array.keys()}
    for method_name, stepbatch_costperiter_array in method_to_stepbatch_costperiter_array.items():
        method_to_stepbatch_array[method_name] = vectorized_first_idx_below_eps(eps, opt_cost, stepbatch_costperiter_array, trunc_val = trunc_val)
    return method_to_stepbatch_array


def method_4_solve_interpolation(A, b, x, minibatch, step, obj, grad):
    # Necoara paper
    delta = 1
    
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]
    subset_obj = [obj(A[[i]], b[[i]], x) for i in indices]

    Lk_N_numerator = np.linalg.norm(np.sum([subset_obj[i] / (np.linalg.norm(subset_grad[i]) ** 2) * subset_grad[i] for i in range(minibatch)], axis = 0))**2 / minibatch ** 2
    Lk_N_denominator = np.sum([subset_obj[i] ** 2 / (np.linalg.norm(subset_grad[i]) ** 2) for i in range(minibatch)]) / minibatch
    
    if Lk_N_numerator == 0:
        return x
    
    Lk_N_inv = Lk_N_denominator /  Lk_N_numerator
    beta_k = (2- delta) * Lk_N_inv 
    update = 0
    for i in range(minibatch):
        update += beta_k * subset_obj[i]/(np.inner(subset_grad[i],subset_grad[i]))*subset_grad[i]
  
    update = update/minibatch

    return x - step * update


def method_1_func_eval(lmbda,G,F,step):
    inn_prod = G@lmbda
    return step/2*np.inner(inn_prod,inn_prod) - np.inner(F,lmbda)

def method_1_func_prime(lmbda,G,F,step):
    return step*G.T@G@lmbda - F

def method_SGD_solve(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]

    avg_grad = np.average(np.vstack(subset_grad), axis = 0)
    return x - step * avg_grad

def method_1_solve_slow(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]
    subset_obj = [obj(A[[i]], b[[i]], x) for i in indices]

    xcvx = cp.Variable(x.size)
    
    first_term = cp.sum([cp.pos(subset_obj[i] + subset_grad[i] @ (xcvx - x)) for i in range(len(subset_grad))])/minibatch
    obj = cp.Minimize(first_term + 1 / (2 * step) * cp.norm(xcvx - x) ** 2)
    cons = []
    prob = cp.Problem(obj, cons)
    prob.solve()
    return xcvx.value

def method_1_solve_fast(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    sub_b = b[indices]
    F = np.array([obj(A[[i]], b[[i]], x) for i in indices])
    G = [grad(A[[i]], b[[i]], x) for i in indices]
    G = np.vstack(G).T
    lmbda,f,d = fmin_l_bfgs_b(method_1_func_eval,1/minibatch*np.ones(minibatch),fprime = method_1_func_prime, args = (G,F,step) ,bounds = [(0,1/minibatch) for i in range(minibatch)])
    return x - step*G@lmbda

def method_1_solve_kinda_fast(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    sub_b = b[indices]
    F = np.array([obj(A[[i]], b[[i]], x) for i in indices])
    G = [grad(A[[i]], b[[i]], x) for i in indices]
    G = np.vstack(G).T
    
    lmbda = cp.Variable(minibatch)
    obj = cp.Maximize(-step/2*cp.sum_squares(G@lmbda) + F@lmbda)
    constr = [0 <= lmbda, lmbda <= 1/minibatch]
    prob = cp.Problem(obj,constr)
    prob.solve()

    return x - step*G@lmbda.value    

def method_1_2_bridge_solve(A, b, x, minibatch, step, obj, grad, beta):
    indices = np.random.randint(A.shape[0], size=minibatch)
    # size of dual
    dual_size = int(minibatch/beta)
    assert minibatch%beta == 0
    # indices divided such that rows will now be averaged.
    index_mat = [indices[i:i + beta] for i in range(0, minibatch, beta)]
    
    F = []
    G = []
    for j in range(len(index_mat)):
        subset_grad = [grad(A[[i]], b[[i]], x) for i in index_mat[j]]
        subset_obj = [obj(A[[i]], b[[i]], x) for i in index_mat[j]]

        G.append(np.average(np.vstack(subset_grad), axis = 0))
        F.append(np.average(subset_obj))
    
    F = np.array(F)
    G = np.vstack(G).T
    
    #if dual_size == 1:
    #    return x - min(step,F[0]/(np.inner(G[0],G[0])))*G[0]
    
    #print(F)
    #print(G)
    #assert len(F) == len(G)
    lmbda,f,d = fmin_l_bfgs_b(method_1_func_eval,1/dual_size*np.ones(dual_size),fprime = method_1_func_prime, args = (G,F,step) ,bounds = [(0,1/dual_size) for i in range(dual_size)])
    return x - step*G@lmbda


def method_2_solve(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]
    subset_obj = [obj(A[[i]], b[[i]], x) for i in indices]

    avg_grad = np.average(np.vstack(subset_grad), axis = 0)
    avg_obj = np.average(subset_obj)

    return x - min(step,avg_obj/(np.inner(avg_grad,avg_grad)))*avg_grad

def method_3_solve(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]
    subset_obj = [obj(A[[i]], b[[i]], x) for i in indices]

    update = 0

    for i in range(minibatch):
        update += min(step,subset_obj[i]/(np.inner(subset_grad[i],subset_grad[i])))*subset_grad[i]
  
    update = update/minibatch

    return x - update

def local_method_3_solve(A, b, x, minibatch, num_local_steps, step, obj, grad):
    loc_arr = np.tile(x.flatten(), (minibatch, 1))
    # each row of loc_arr is x
    for k in range(num_local_steps):
        indices = np.random.randint(A.shape[0], size=minibatch)
        subset_grad = [grad(A[[i]], b[[i]], loc_arr[j]) for j, i in enumerate(indices)]
        subset_obj = [obj(A[[i]], b[[i]], loc_arr[j]) for j, i in enumerate(indices)]
        for l in range(minibatch):
            update = min(step,subset_obj[l]/(np.inner(subset_grad[l],subset_grad[l])))*subset_grad[l]
            loc_arr[l] = loc_arr[l] - update
    return np.average(loc_arr, axis=0)

def method_5_solve(A, b, x, minibatch, step, obj, grad):
    # Necoara Paper method ?likely with a polyak step size type thing?
    indices = np.random.randint(A.shape[0], size=minibatch)
    subset_grad = [grad(A[[i]], b[[i]], x) for i in indices]
    subset_obj = [obj(A[[i]], b[[i]], x) for i in indices]
    
    Lk_N_numerator = np.linalg.norm(np.sum([subset_obj[i] / (np.linalg.norm(subset_grad[i]) ** 2) * subset_grad[i] for i in range(minibatch)], axis = 0))**2 / minibatch ** 2
    Lk_N_denominator = np.sum([subset_obj[i] ** 2 / (np.linalg.norm(subset_grad[i]) ** 2) for i in range(minibatch)]) / minibatch
    
    if Lk_N_numerator == 0:
        left_term = np.inf
    else:
        left_term = step * Lk_N_denominator / Lk_N_numerator
    
    update = 0

    for i in range(minibatch):
        update += min(left_term, 1) * subset_obj[i]/(np.inner(subset_grad[i],subset_grad[i])) *subset_grad[i]
  
    update = update/minibatch

    return x - update


def method_full_prox_solve_regression(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    invers = np.linalg.solve(minibatch * np.eye(minibatch) + step*sub_A@sub_A.T,sub_A@x - b[indices])
    return x - step*sub_A.T@invers

def method_full_prox_solve_phase(A, b, x, minibatch, step, obj, grad):
    # Assuming b >= 0, sets x to be the minimizer of
    #
    #   |(a'*x)^2 - b| + ||x - x_init||^2 / (2 * alpha).
    #
    # By a subgradient calculation, the vector x must be of one of the following
    # three forms (the first corresponds to projection onto the set (a'*x)^2 = b):
    #
    # x = (I - a*a' / norm(a)^2) * x_init
    #      + sign(a' * x_init) * a * sqrt(b) / norm(a)^2
    #
    # x = (I - 2 * alpha * a * a') \ x_init
    #   = (I + 2 * alpha * a * a' / (1 + 2 * alpha * norm(a)^2)) * x_init
    #
    # x = (I + 2 * alpha * a * a') \ x_init
    #   = (I - 2 * alpha * a * a' / (1 + 2 * alpha * norm(a)^2)) * x_init

#     indices = np.random.randint(A.shape[0], size=minibatch)
#     sub_A = A[indices]
#     invers = np.linalg.solve(minibatch * np.eye(minibatch) + step*sub_A@sub_A.T,sub_A@x - b[indices])
    return x

def method_full_prox_l1_regression_func_eval(v, A, b, x, minibatch,step):
    #v is the variable
    ATv = A.T @ v
    return step/ (2 * minibatch ** 2) * np.inner(ATv, ATv) - np.inner(ATv, x)/ minibatch + np.inner(v, b)/minibatch

def method_full_prox_l1_regression_func_prime(v, A, b, x, minibatch,step):
    return step/(minibatch ** 2) * A @ (A.T @ v) - A@x / minibatch + b / minibatch

def method_full_prox_solve_l1_regression(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    sub_b = b[indices]
    v,f,d = fmin_l_bfgs_b(method_full_prox_l1_regression_func_eval,1/minibatch*np.ones(minibatch),fprime = method_full_prox_l1_regression_func_prime, args = (sub_A, sub_b, x, minibatch,step) ,bounds = [(-1,1) for i in range(minibatch)])
    return x - step*sub_A.T@v/minibatch

def method_full_prox_log_regression_func_eval(x, A, b, xk, minibatch,step):
    #x is the variable
    return np.sum(np.log(1 + np.exp(-b * (A@x)))) / minibatch + 1/(2 * step) * np.linalg.norm(x- xk)**2

def method_full_prox_log_regression_func_prime(x, A, b, xk, minibatch,step):
    exp_bAx = np.exp(b * (A@x))
    coeff = (1 / (1 + exp_bAx)) * (-b)
    return A.T @  coeff + x / step - xk / step

def method_full_prox_solve_log_regression(A, b, x, minibatch, step, obj, grad):
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    sub_b = b[indices]
    newx,f,d = fmin_l_bfgs_b(method_full_prox_log_regression_func_eval,np.ones(x.shape),fprime = method_full_prox_log_regression_func_prime, args = (sub_A, sub_b, x, minibatch,step))
    return newx

def method_full_prox_projection_func_eval(lmbda,G,F,step):
    inn_prod = G@lmbda
    return step/2*np.inner(inn_prod,inn_prod) - np.inner(F,lmbda)

def method_full_prox_projection_func_prime(lmbda,G,F,step):
    return step*G.T@G@lmbda - F


def method_full_prox_solve_projection(A, b, x, minibatch, step, obj, grad):
    # obj, grad not used
    # using method 1 solve code with customized F and G to make it prox solve for projection problem
    indices = np.random.randint(A.shape[0], size=minibatch)
    sub_A = A[indices]
    sub_b = b[indices]
    F = np.array([(np.inner(A[i],x)-b[i])/np.linalg.norm(A[i]) for i in indices])
    G = [A[i]/np.linalg.norm(A[i])for i in indices]
    G = np.vstack(G).T
    lmbda,f,d = fmin_l_bfgs_b(method_full_prox_projection_func_eval,1/minibatch*np.ones(minibatch),fprime = method_full_prox_projection_func_prime, args = (G,F,step) ,bounds = [(0,1/minibatch) for i in range(minibatch)])
    return x - step*G@lmbda


def l1_regression_grad(A, b, x):
    # return needs to be flattened
    assert A.shape[0] == 1
    return (np.sign(A@x -b) * A).flatten()
    
def l1_regression_obj(A, b, x):
    return np.linalg.norm(A @ x - b, ord=1)

def regression_aprox_method_max_samples(A, b, minibatch, alpha, solve, seed, obj, grad, max_sample = 1000, verbose = False):
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    
    max_iteration = np.int(np.ceil(max_sample / minibatch))
    
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        step = alpha / np.sqrt(iteration + 1)
        x = solve(A, b, x, minibatch, step, obj, grad)
    cost_lst.append(obj(A, b, x) / n)
    if not verbose:
        return np.array(cost_lst)
    else:
        return np.array(cost_lst), x

    
def regression_aprox_method_max_samples_acc(A, b, minibatch, gamma_0, solve, seed, obj, grad, max_sample = 1000, verbose = False, beta_0 = 1):
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    x_ag = np.copy(x)
    
    max_iteration = np.int(np.ceil(max_sample / minibatch))
    
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        beta = beta_0 * np.sqrt(iteration + 2) / 2
        gamma = gamma_0 * np.sqrt(iteration + 2) / 2

        x_md = x / beta + (1 - 1/beta) * x_ag
        x = solve(A, b, x_md, minibatch, gamma, obj, grad)
        x_ag = x / beta + (1 - 1/beta) * x_ag
    cost_lst.append(obj(A, b, x) / n)
    if not verbose:
        return np.array(cost_lst)
    else:
        return np.array(cost_lst), x
    
def regression_aprox_method(A, b, minibatch, alpha, solve, seed, obj, grad, max_iteration = 1000, verbose = False):
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        step = alpha / np.sqrt(iteration + 1)
        x = solve(A, b, x, minibatch, step, obj, grad)
    cost_lst.append(obj(A, b, x) / n)
    if not verbose:
        return np.array(cost_lst)
    else:
        return np.array(cost_lst), x

def regression_aprox_method_time_to_eps(A, b, eps, opt_cost, minibatch, alpha, solve, seed, obj, grad, max_iteration = 1000):
    # returns the iterations needed to get 
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        if np.abs(cost_lst[-1] - opt_cost) < eps:
            return iteration + 1
        step = alpha / np.sqrt(iteration + 1)
        x = solve(A, b, x, minibatch, step, obj, grad)
    cost_lst.append(obj(A, b, x) / n)
    return np.inf

def regression_aprox_method_time_to_eps_local_method3(A, b, eps, opt_cost, minibatch, num_local_steps, alpha, solve, seed, obj, grad, max_iteration = 1000):
    # returns the iterations needed to get 
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        if np.abs(cost_lst[-1] - opt_cost) < eps:
            return iteration + 1
        step = alpha / np.sqrt(iteration + 1)
        x = solve(A, b, x, minibatch, num_local_steps, step, obj, grad)
    cost_lst.append(obj(A, b, x) / n)
    return np.inf

def regression_aprox_method_time_to_eps_accel(A, b, eps, opt_cost, minibatch, gamma_0, solve, seed, obj, grad, max_iteration = 1000, beta_0=1):
    # returns the iterations needed to get 
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    x_ag = np.copy(x)
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        if np.abs(cost_lst[-1] - opt_cost) < eps:
            return iteration + 1

        beta = beta_0 * np.sqrt(iteration + 2) / 2
        gamma = gamma_0 * np.sqrt(iteration + 2) / 2

        x_md = x / beta + (1 - 1/beta) * x_ag
        x = solve(A, b, x_md, minibatch, gamma, obj, grad)
        x_ag = x / beta + (1 - 1/beta) * x_ag
    cost_lst.append(obj(A, b, x) / n)
    return np.inf

def regression_aprox_method_time_to_eps_bridge(A, b, eps, opt_cost, minibatch, alpha, solve, seed, obj, grad, beta, max_iteration = 1000):
    # returns the iterations needed to get 
    np.random.seed(seed)
    cost_lst = []
    n, d = A.shape
    x = np.random.normal(0, 1, d)
    for iteration in range(max_iteration):
        cost_lst.append(obj(A, b, x) / n)
        if np.abs(cost_lst[-1] - opt_cost) < eps:
            return iteration + 1
        step = alpha / np.sqrt(iteration + 1)
        x = solve(A, b, x, minibatch, step, obj, grad,beta)
    cost_lst.append(obj(A, b, x) / n)
    return np.inf

def generate_data(n,d,source,noise_param,ran_state,condition,regression,obj):
    np.random.seed(ran_state)
    if regression == 'phase':
        assert noise_param is None or noise_param == 0
        A = np.random.randn(n,d) 
        A = A / np.linalg.norm(A, axis=1).reshape((-1, 1))
        
        x = np.random.randn(d);
        noise = np.abs(noise_param * np.random.randn(n))
        b = (A @ x) ** 2 
        opt_cost = 0
        
    if regression == 'linear':
        if source == 'sklearn':
            A, b = sklearn.datasets.make_regression(n, d, random_state=None, noise = noise_param)
        
        elif source == 'self':
            A = np.random.randn(n,d)

        if condition > 0:
            x_true = np.random.randn(d)
            Q , R = np.linalg.qr(A)
            A = np.sqrt(n) * Q @ np.diag(np.linspace(1, condition, d))
            b = A@x_true + noise_param*np.random.randn(n)
        
        if noise_param > 0: 
            true_x, _, _, _ = np.linalg.lstsq(A, b)
            opt_cost = obj(A, b, true_x) / n
        else:
            opt_cost = 0    

    
    if regression == 'abs':
        
        if source == 'sklearn':
            A, b = sklearn.datasets.make_regression(n, d, random_state=None, noise = noise_param)
        
        elif source == 'self':
            A = np.random.randn(n,d)

        if condition > 0:
            x_true = np.random.randn(d)
            Q , R = np.linalg.qr(A)
            A = np.sqrt(n) * Q @ np.diag(np.linspace(1, condition, d))
            b = A@x_true + np.random.laplace(loc = 0, scale = noise_param/np.sqrt(2),size = n)
        
        if noise_param > 0: 
            true_x = cp.Variable(A.shape[1])
            objective = cp.Minimize(cp.norm(A @ true_x - b, 1))
            cp.Problem(objective, []).solve(solver=cp.MOSEK)
            opt_cost = obj(A, b, true_x.value) / n

        else:
            opt_cost = 0

    if regression == 'log':
        
        A = np.random.randn(n,d)

        if condition > 0:
            x_true = np.random.randn(d)
            Q , R = np.linalg.qr(A)
            A = np.sqrt(n) * Q @ np.diag(np.linspace(1, condition, d))
        
        x = np.random.randn(d);
        b = np.sign(A @ x);

        if noise_param > 0:
            dummy = np.random.random(n)
            b[dummy < noise_param] = -1*b[dummy<noise_param]

        true_x = cp.Variable(A.shape[1])
        objective = cp.Minimize(cp.sum(cp.logistic(-cp.multiply(b, A@true_x))))
        opt_cost = cp.Problem(objective, []).solve(solver=cp.MOSEK) / n

    if regression == 'projection':
        A = np.random.randn(n,d)

        if condition > 0:
            Q , R = np.linalg.qr(A)
            A = np.sqrt(n) * Q @ np.diag(np.linspace(1, condition, d))
        
        x = np.random.randn(d);
        noise = np.abs(noise_param * np.random.randn(n))
        b = A @ x + noise
        
        cvx_x = cp.Variable(d)
        objective = cp.Minimize(0)
        prob = cp.Problem(objective, [A @ cvx_x <= b])
        val = prob.solve(solver=cp.MOSEK)
        assert prob.status != "infeasible" and prob.status != "unbounded"
        assert val == 0.0
        opt_cost = 0
    return A,b,opt_cost    