# Dean Li 
# CPSC 490: Senior Project
# There are two files: bias_functions.py contains the helpful functions,
# and bias_main.py contains the simulations. We run different simulations
# of the multi-choice secretary problem with bias. This file contains the simulations.

import random
import time
from st_functions import *
import csv
from functools import partial

# This function takes in a list of bias levels, a p, and the number of simulations
# that we would like to run and returns the simulated probability of success for the
# utopian group-blind algorithm.
def utopianBiasArray(biases, percent, num):

    bias_results = [0] * len(biases)

    for i in range(0, num):
        temp, saved_order = true_arrivals(10000, percent)
        untemp = zip(*temp)
        candidates, groups = list(untemp)

        for z in range(0, len(bounds)):
            if bounds[z] == 1:
                bias_results[z] += nobias(candidates, groups, 7)
            else: 
                true_bound = goodBound(saved_order, bounds[z])
                bias_results[z] += low_bias(candidates, groups, 7, true_bound)

    
    results = [x / num for x in bias_results]
    return results

# This function takes in the variable T.
# It will construct the optimal relationship between the T and C that we arrived at
# and empirically test the strategy to arrive at the simulated success percentage.
# Notice that t_frac is last: this is so we can use partial from functools for
# Bayesian optimization. 
def window_test(p, q, k, j, num, t_frac):
    T = -1 * math.log(t_frac)
    if q != 0:
        C = math.log(pq / (1 - (p * q) * (j - 1) / j)) + T
    else:
        C = 0

    if math.exp(-1 * C) < 1:
        win_bias = dynamic_bound_bias(p, q, k, j, t_frac, num)[0]
        win_nobias = dynamic_bound_nobias(p, q, k, j, t_frac, num)[0]

        return q * win_bias + (1 - q) * win_nobias

    else:
        win_bias = 0
        win_nobias = 0

        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            true_bound = goodBound(saved_order, j)
            win_bias += low_biasT(candidates, groups, T, k, true_bound)
            win_nobias += nobiasT(candidates, groups, T, k)

        win_bias = win_bias / num
        win_nobias = win_nobias / num

        return q * win_bias + (1 - q) * win_nobias

# We can directly simulate the success of the window mechanism in the no bias scenario
# Notice the inverted order of T and the number of simulations. This returns the error
# rate on minority and majority candidates as the second and third item in the list.
def window_nobias(p, q, k, j, t_frac, num):
    
    T = -1 * math.log(t_frac)
    if q != 0:
        C = math.log(p * q / (1 - (p * q) * (j - 1) / j)) + T
    else:
        C = 0

    success = 0
    min_success = 0
    min_num = 0

    if math.exp(-1 * C) < 1:
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            x = window_nobias(candidates, groups, T, C, k)
            success += x
            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

    else:
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)


            x = nobiasT(candidates, groups, T, k)
            success += x
            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

# We can directly simulate the success of the window mechanism in the biased scenario.
# This returns the error rate on minority and majority candidates as the second 
# and third item in the list.
def dynamic_bound_bias(p, q, k, j, t_frac, num):

    T = -1 * math.log(t_frac)
    if q != 0:
        C = math.log(p * q / (1 - (p * q) * (j - 1) / j)) + T
    else:
        C = 0

    success = 0
    min_success = 0
    min_num = 0

    if math.exp(-1 * C) < 1:
        start_time = time.time()
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            true_bound = goodBound(saved_order, j)
            x = window_bias(candidates, groups, T, C, k, true_bound)
            success += x

            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

            if i % 1000 == 0:
                print(i / num)
                print("--- %s seconds ---" % (time.time() - start_time))
                if i != 0:
                    print(success / i)

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

    else:
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            true_bound = goodBound(saved_order, j)
            x = low_biasT(candidates, groups, T, k, true_bound)
            success += x

            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

# We can directly simulate the success of the window mechanism in the biased scenario.
# We now SPECIFY our C, and this allows us to do the pre-calibrated mechanism evaluation
# as we do in Section 5. As above, this returns minority and majority error rates.
def dynamic_bound_bias_C(p, q, k, j, t_frac, C, num):

    T = -1 * math.log(t_frac)
    if q != 0:
        C = -1 * math.log(C)
    else:
        C = 0

    success = 0
    min_success = 0
    min_num = 0

    if C < 1:
        start_time = time.time()
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            true_bound = goodBound(saved_order, j)
            x = window_bias(candidates, groups, T, C, k, true_bound)
            success += x

            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

            if i % 1000 == 0:
                print(i / num)
                print("--- %s seconds ---" % (time.time() - start_time))
                if i != 0:
                    print(success / i)

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

    else:
        for i in range(0, num):
            temp, saved_order = true_arrivals(10000, p)
            untemp = zip(*temp)
            candidates, groups = list(untemp)

            true_bound = goodBound(saved_order, j)
            x = low_biasT(candidates, groups, T, k, true_bound)
            success += x

            if saved_order[0] == 1:
                min_num += 1 
                min_success += x

        return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

# This is a specific case of the utopianBias function that also returns minority and majority error rates.
def utopianBias(p, k, j, num):
    success = 0
    min_success = 0
    min_num = 0

    for i in range(0, num):
        temp, saved_order = true_arrivals(10000, p)
        untemp = zip(*temp)
        candidates, groups = list(untemp)

        true_bound = goodBound(saved_order, j)
        x = low_bias(candidates, groups, k, true_bound)
        success += x

        if saved_order[0] == 1:
            min_num += 1 
            min_success += x


    return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]


# We directly simulate the success of the designated slot mechanism in the no bias scenario
# We specify T directly. This returns minority and majority error rates.
def slot_nobias(p, q, k, j, t_frac, num):
    T = math.factorial(k - 1) ** (1 / (k - 1))

    res = minimize(func1, 1, args=(1, j))
    C = res.x[0]

    success = 0
    min_success = 0
    min_num = 0
    
    for i in range(0, num):
        temp, saved_order = true_arrivals(10000, p)
        untemp = zip(*temp)
        candidates, groups = list(untemp)

        x = Rooney_nobias(candidates, groups, T, C, k)
        success += x

        if saved_order[0] == 1:
            min_num += 1 
            min_success += x

    return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

# We directly simulate the success of the designated slot mechanism in the biased scenario
# We specify T directly. this returns minority and majority error rates.
def slot_bias(p, q, k, j, t_frac, num):

    T = math.factorial(k - 1) ** (1 / (k - 1))

    res = minimize(func1, 1, args=(1, j))
    C = res.x[0]

    success = 0
    min_success = 0
    min_num = 0
    
    for i in range(0, num):
        temp, saved_order = true_arrivals(10000, p)
        untemp = zip(*temp)
        candidates, groups = list(untemp)

        true_bound = goodBound(saved_order, j)
        x = Rooney_bias(candidates, groups, T, C, k, true_bound)
        success += x

        if saved_order[0] == 1:
            min_num += 1 
            min_success += x

    return [success / num, 1 - min_success / min_num, 1 - (success - min_success) / (num - min_num)]

# This function uses the upper bound for group-blind algorithms and the lower bound for our 
# designated slot mechanism to calculate the q necessary for our designated slot mechanism to
# outperform the group-blind algorithm across different p values. We calculate this q to the 
# nearest .005.
def find_intersection(k, z, j):
    q_answers = []

    q_last = 1
    for j in range(0, 100, 1):
        p = j / 200
        
        if lowerBoundC(p, 1, k, z, j) <= upperBoundT(p, 1, k, j):
            q_answers.append(1.01)
            continue

        else:
            for i in range(1, 1000):
                q = q_last - i / 1000
                if lowerBoundC(p, q, k, z, j) <= upperBoundT(p, q, k, j):
                    q_answers.append(q + .001)
                    q_last = q + .001
                    break
                    
    return q_answers

# We use this function for our Bayesian optimization, where num represents the number of simulations
# we would like to run for each function evaluation. We evaluate at q with intervals of 0.05.
def optimize_T(p, k, j, num):
    answers = []
    for i in range(0, 1050, 50):
        q = i / 1000

        box = partial(dynamic_bound_test, p, q, k, j, num)
        best_guess = math.exp(-1 * math.factorial(k) ** (1 / k))

        pbounds = {'t_frac': (best_guess / 2, best_guess * 2)}

        optimizer = BayesianOptimization(
            f=box,
            pbounds=pbounds,
            random_state=1
        )

        optimizer.probe(
            params={"t_frac": math.exp(-1 * math.factorial(k) ** (1 / k))},
            lazy=True
        )

        optimizer.probe(
            params={"t_frac": math.exp(-1 * math.factorial(k) ** (1 / k)) * k / (k + 1)},
            lazy=True
        )

        optimizer.probe(
            params={"t_frac": math.exp(-1 * math.factorial(k) ** (1 / k)) * (k  + 1) / k},
            lazy=True
        )

        optimizer.maximize(
            init_points=2,
            n_iter=20,
            alpha=2.5e-3
        )

        answers.append(optimizer.max['params']['t_frac'])

    return answers
