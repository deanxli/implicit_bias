# Dean Li 
# CPSC 490: Senior Project
# There are two files: bias_functions.py contains the helpful functions,
# and bias_main.py contains the simulations. We run different simulations
# of the multi-choice secretary problem with bias. This file contains the functions. 

import random
import math
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization

######################################################################################
############################    BASIC  FUNCTIONS    ##################################
######################################################################################

# Returns the order of the group and the distribution of ranks which can be used 
# in conjuction with goodBound to simulate bias
def true_arrivals(number, p):
    ranking = list(range(0, number))
    group = list([0] * round((1 - p) * number) + [1] * round(p * number))

    random.shuffle(group)
    saved_order = group

    candidate = list(zip(ranking, group))
    random.shuffle(candidate)
    return candidate, saved_order

# We want to return the first index after which we will have seen the appropriate number of
# majority candidates -- the bound refers to the j^th best: for example, a bound of five
# means that no minority candidate will appear higher than the 5^{th} best candidate, meaning
# there are four majority candidates above them. 
def goodBound(saved_order, bound):
    i = 0
    while bound > 1 and i < len(saved_order):
        if saved_order[i] == 0:
            bound -= 1
        i += 1
    return i

######################################################################################
############################    TRIAL  FUNCTIONS    ##################################
######################################################################################

# A no bias trial -- just to confirm the empirical results. The exploration period is
# set automatically to the optimal size for the no bias situation.
def nobias(candidates, groups, k):
    T = round(len(candidates) * math.exp(-1 * (math.factorial(k) ** (1. / k))))
    
    best = min(candidates[:T])
    if best == 0:
        return 0
    
    else: 
        counter = k
        min_candidates = 0
        i = T
        while counter > 0 and i < len(candidates):
            if candidates[i] == 0:
                return 1
            elif candidates[i] < best:
                best = candidates[i]
                counter -= 1
            i += 1
            
        return 0

# This runs the "lowest bias" setting for the ceiling:
# all minority candidates above the ceiling are dropped right below and
# their order is preserved. For example, if the 3rd and 5th best were 
# minority candidates and the ceiling were 10, now they would be 11th and 12th best.
# The ceiling value used should be from goodBound 
def low_bias(candidates, groups, k, ceiling):
    
    T = round(len(candidates) * math.exp(-1 * (math.factorial(k) ** (1. / k))))
    
    best = min(candidates[:T])
    # If the best is in the already chosen candidates, we end the function
    if best == 0:
        return 0
    
    # Otherwise, we begin our search
    else: 
        counter = k
        true_minority_rank = 0
        i = T

        while counter > 0 and i < len(candidates):
            # Handling the minority candidates
            if groups[i] == 1:
                # If the ceiling applies and we should go into action
                if candidates[i] < ceiling:
                    if ceiling < best:
                        best = ceiling
                        counter -= 1
                        true_minority_rank = candidates[i]
                        # We've selected the best!
                        if candidates[i] == 0:
                            return 1
                    elif ceiling == best:
                        if candidates[i] < true_minority_rank:
                            counter -= 1
                            true_minority_rank = candidates[i]
                            # We've selected the best!
                            if candidates[i] == 0:
                                return 1

                # The ceiling doesn't apply!
                else:
                    if candidates[i] < best:
                        best = candidates[i]
                        counter -= 1
                i += 1        
            
            # Handling the majority candidates
            else:
                if candidates[i] == 0:
                    return 1
                elif candidates[i] < best:
                    best = candidates[i]
                    counter -= 1
                i += 1
            
        return 0


# The low bias trial but we are allowed to specify the size of the exploration period.
# Note: the exploration period size takes the form e^-a.
def low_biasT(candidates, groups, T, k, ceiling):
    T = round(len(candidates) * math.exp(-1 * T))
    
    best = min(candidates[:T])
    # If the best is in the already chosen candidates, we end the function
    if best == 0:
        return 0
    
    # Otherwise, we begin our search
    else: 
        counter = k
        true_minority_rank = 0
        i = T

        while counter > 0 and i < len(candidates):
            # Handling the minority candidates
            if groups[i] == 1:
                # If the ceiling applies and we should go into action
                if candidates[i] < ceiling:
                    if ceiling < best:
                        best = ceiling
                        counter -= 1
                        true_minority_rank = candidates[i]
                        # We've selected the best!
                        if candidates[i] == 0:
                            return 1
                    elif ceiling == best:
                        if candidates[i] < true_minority_rank:
                            counter -= 1
                            true_minority_rank = candidates[i]
                            # We've selected the best!
                            if candidates[i] == 0:
                                return 1

                # The ceiling doesn't apply!
                else:
                    if candidates[i] < best:
                        best = candidates[i]
                        counter -= 1
                i += 1        
            
            # Handling the majority candidates
            else:
                if candidates[i] == 0:
                    return 1
                elif candidates[i] < best:
                    best = candidates[i]
                    counter -= 1
                i += 1
            
        return 0

# The no bias trial but we are allowed to specify the size of the exploration period.
# Note: the exploration period size takes the form e^-a.
def nobiasT(candidates, groups, T, k):
    T = round(len(candidates) * math.exp(-1 * T))
    
    best = min(candidates[:T])
    if best == 0:
        return 0
    
    else: 
        counter = k
        min_candidates = 0
        i = T
        while counter > 0 and i < len(candidates):
            if candidates[i] == 0:
                return 1
            elif candidates[i] < best:
                best = candidates[i]
                counter -= 1
            i += 1
            
        return 0

# A simulation of the window mechanism where we specify the T and Tc analogues. There
# is bias in this environment.
def window_bias(candidates, groups, e_T, e_C, k, ceiling):
    min_candidates = np.array(candidates)[np.array(np.nonzero(groups))].tolist()[0]

    T = round(math.exp(-1 * e_T) * len(candidates))
    C = round(math.exp(-1 * e_C) * len(min_candidates))

    best_T = min(candidates[:T])
    best_C = min(min_candidates[:C])
    counter = k

    # If the best is in the already chosen candidates, we end the function
    if best_T == 0 and best_C == 0:
        return 0
    
    # Otherwise, we begin our search from the beginning: we should choose
    # where we start from whichever threshold is hit first
    # tFlag and cFlag represent if we are using certain selection criteria
    min_seen = 0
    best = best_min = 0

    if sum(groups[:T]) <= C:
            # We can begin selecting candidates with T, we still have to count 
            # down until we see enough minority candidates however
            i = T
            min_seen = sum(groups[:T])
            tFlag = True
            best = best_T
            best_min = min(min_candidates[:sum(groups[:T])])

    else:
            # We can begin selecting candidates with C, we have to count down T
            i = candidates.index(min_candidates[C])
            min_seen = C
            best = min(candidates[:i])
            best_min = best_C

    while counter > 0 and i < len(candidates):
            selected = False
            if i >= T:
                # Handling the minority candidates
                if groups[i] == 1:
                    # If the ceiling applies and we should go into action
                    if candidates[i] < ceiling:
                        if ceiling < best:
                            best = ceiling
                            counter -= 1
                            selected = True
                            # We've selected the best!
                            if candidates[i] == 0:
                                return 1
                        elif ceiling == best:
                            if candidates[i] < best_min:
                                counter -= 1
                                selected = True
                                # We've selected the best!
                                if candidates[i] == 0:
                                    return 1

                    # The ceiling doesn't apply!
                    else:
                        if candidates[i] < best:
                            best = candidates[i]
                            counter -= 1
                            selected = True

                    best_min = min(candidates[i], best_min)
                    i += 1
                    min_seen += 1      
                
                # Handling the majority candidates
                else:
                    if candidates[i] == 0:
                        return 1
                    elif candidates[i] < best:
                        best = candidates[i]
                        counter -= 1
                    i += 1
            
                if min_seen >= C and not selected and groups[i - 1] == 1:
                    if candidates[i - 1] <= best_min:
                        counter -= 1
                        best_min = min(best_min, candidates[i - 1])
                        if candidates[i - 1] == 0:
                            return 1
            else:
                if min_seen >= C and not selected and groups[i] == 1:
                    if candidates[i] <= best_min:
                        counter -= 1
                        best_min = min(best_min, candidates[i])
                        if candidates[i] == 0:
                            return 1
                i += 1

    return 0

# A simulation of the window mechanism where we specify the T and Tc analogues. There
# is no bias in this environment.
def window_nobias(candidates, groups, e_T, e_C, k):
    min_candidates = np.array(candidates)[np.array(np.nonzero(groups))].tolist()[0]

    T = round(math.exp(-1 * e_T) * len(candidates))
    C = round(math.exp(-1 * e_C) * len(min_candidates))
    print(T)
    print(C)

    best_T = min(candidates[:T])
    best_C = min(min_candidates[:C])
    counter = k

    # If the best is in the already chosen candidates, we end the function
    if best_T == 0 and best_C == 0:
        return 0
    
    # Otherwise, we begin our search from the beginning: we should choose
    # where we start from whichever threshold is hit first
    # tFlag and cFlag represent if we are using certain selection criteria
    min_seen = 0
    best = best_min = 0

    if sum(groups[:T]) <= C:
            # We can begin selecting candidates with T, we still have to count 
            # down until we see enough minority candidates however
            i = T
            min_seen = sum(groups[:T])
            tFlag = True
            best = best_T
            best_min = min(min_candidates[:sum(groups[:T])])

    else:
        # We can begin selecting candidates with C, we have to count down T
        i = candidates.index(min_candidates[C])
        min_seen = C
        best = min(candidates[:i])
        best_min = best_C

    while counter > 0 and i < len(candidates):
        selected = False
        if i >= T:
            if candidates[i] == 0:
                return 1
            elif candidates[i] < best:
                best = candidates[i]
                selected = True
                counter -= 1

            if groups[i] == 1:
                best_min = min(best_min, candidates[i])
                min_seen += 1
            
            i += 1
        
            if min_seen >= C and not selected and groups[i - 1] == 1:
                if candidates[i - 1] <= best_min:
                    counter -= 1
                    best_min = min(best_min, candidates[i - 1])
                    if candidates[i - 1] == 0:
                        return 1
        else:
            if min_seen >= C and not selected and groups[i] == 1:
                if candidates[i] <= best_min:
                    counter -= 1
                    best_min = min(best_min, candidates[i])
                    if candidates[i] == 0:
                        return 1
            i += 1

    return 0

# A simulation of the designated slot mechanism where we specify the T and C analogues. 
# There is bias in this environment.
def Rooney_bias(candidates, groups, e_T, e_C, k, ceiling):
    min_candidates = np.array(candidates)[np.array(np.nonzero(groups))].tolist()[0]

    T = round(math.exp(-1 * e_T) * len(candidates))
    C = round(math.exp(-1 * e_C) * len(min_candidates))

    best_T = min(candidates[:T])
    best_C = min(min_candidates[:C])
    counter = k - 1
    min_counter = 1

    # If the best is in the already chosen candidates, we end the function
    if best_T == 0 and best_C == 0:
        return 0
    
    # Otherwise, we begin our search from the beginning: we should choose
    # where we start from whichever threshold is hit first
    # tFlag and cFlag represent if we are using certain selection criteria
    min_seen = 0
    best = best_min = 0

    if sum(groups[:T]) <= C:
            # We can begin selecting candidates with T, we still have to count 
            # down until we see enough minority candidates however
            i = T
            min_seen = sum(groups[:T])
            tFlag = True
            best = best_T
            best_min = min(min_candidates[:sum(groups[:T])])

    else:
        # We can begin selecting candidates with C, we have to count down T
        i = candidates.index(min_candidates[C])
        min_seen = C
        best = min(candidates[:i])
        best_min = best_C

    while (counter > 0 or min_counter > 0) and i < len(candidates):
        selected = False

        if i >= T and counter > 0:

            # Handling the minority candidates
            if groups[i] == 1:
                # If the ceiling applies and we should go into action
                if candidates[i] < ceiling:
                    if ceiling < best:
                        best = ceiling
                        counter -= 1
                        selected = True
                        # We've selected the best!
                        if candidates[i] == 0:
                            return 1
                    elif ceiling == best:
                        if candidates[i] < best_min:
                            counter -= 1
                            selected = True
                            # We've selected the best!
                            if candidates[i] == 0:
                                return 1

                # The ceiling doesn't apply!
                else:
                    if candidates[i] < best:
                        best = candidates[i]
                        counter -= 1
                        selected = True

                best_min = min(candidates[i], best_min)
                i += 1
                min_seen += 1      
            
            # Handling the majority candidates
            else:
                if candidates[i] == 0:
                    return 1
                elif candidates[i] < best:
                    best = candidates[i]
                    counter -= 1
                i += 1  

            if min_seen >= C and not selected and groups[i - 1] == 1 and min_counter > 0:
                if candidates[i - 1] <= best_min:
                    min_counter -= 1
                    best_min = min(best_min, candidates[i - 1])
                    if candidates[i - 1] == 0:
                        return 1
        else:
            if min_seen >= C and not selected and groups[i] == 1 and min_counter > 0:
                if candidates[i] <= best_min:
                    min_counter -= 1
                    best_min = min(best_min, candidates[i])
                    if candidates[i] == 0:
                        return 1
            i += 1


    return 0

# A simulation of the designated slot mechanism where we specify the T and C analogues. 
# There is no bias in this environment.
def Rooney_nobias(candidates, groups, e_T, e_C, k):
    min_candidates = np.array(candidates)[np.array(np.nonzero(groups))].tolist()[0]

    T = round(math.exp(-1 * e_T) * len(candidates))
    C = round(math.exp(-1 * e_C) * len(min_candidates))

    best_T = min(candidates[:T])
    best_C = min(min_candidates[:C])
    counter = k - 1
    min_counter = 1

    # If the best is in the already chosen candidates, we end the function
    if best_T == 0 and best_C == 0:
        return 0
    
    # Otherwise, we begin our search from the beginning: we should choose
    # where we start from whichever threshold is hit first
    # tFlag and cFlag represent if we are using certain selection criteria
    min_seen = 0
    best = best_min = 0

    if sum(groups[:T]) <= C:
            # We can begin selecting candidates with T, we still have to count 
            # down until we see enough minority candidates however
            i = T
            min_seen = sum(groups[:T])
            tFlag = True
            best = best_T
            best_min = min(min_candidates[:sum(groups[:T])])

    else:
        # We can begin selecting candidates with C, we have to count down T
        i = candidates.index(min_candidates[C])
        min_seen = C
        best = min(candidates[:i])
        best_min = best_C

    while (counter > 0 or min_counter > 0) and i < len(candidates):
        
        selected = False
        if i >= T and counter > 0:
            if candidates[i] == 0:
                return 1
            elif candidates[i] < best:
                best = candidates[i]
                selected = True
                counter -= 1

            if groups[i] == 1:
                best_min = min(best_min, candidates[i])
                min_seen += 1
            
            i += 1
        
            if min_seen >= C and not selected and groups[i - 1] == 1 and min_counter > 0:
                if candidates[i - 1] <= best_min:
                    min_counter -= 1
                    best_min = min(best_min, candidates[i - 1])
                    if candidates[i - 1] == 0:
                        return 1
        else:
            if min_seen >= C and not selected and groups[i] == 1 and min_counter > 0:
                if candidates[i] <= best_min:
                    min_counter -= 1
                    best_min = min(best_min, candidates[i])
                    if candidates[i] == 0:
                        return 1
            i += 1

    return 0

######################################################################################
###########################   THEORETICAL RESULTS   ##################################
######################################################################################

# This returns the optimal win results for if we split up the two groups completely
def naiveSplit(k, p):
    def sumOptimalFunction(k):
        total = 0
        for i in range(1, k + 1):
            total += ((math.factorial(k) ** (1 / k)) ** i) / (math.factorial(i))

        return total

    best_win = 0
    best_k = 0
    for i in range(0, k + 1):
        if i == 0:
            win = (1 - p) * math.exp(-1 * math.factorial(k) ** (1 / k)) * sumOptimalFunction(k)
        elif i == k:
            win = p * math.exp(-1 * math.factorial(k) ** (1 / k)) * sumOptimalFunction(k)
        else:
            win = p * math.exp(-1 * math.factorial(i) ** (1 / i)) * sumOptimalFunction(i) + \
                (1 - p) * math.exp(-1 * math.factorial(k - i) ** (1 / (k - i))) * sumOptimalFunction(k - i)


        if win > best_win:
            best_win = win
            best_k = i
        else:
            return best_win

    return best_win

# This is the function that we optimize to figure out the C* estimate from Section 4
def func1(c, z, j):
    return -1 * (1 - (1 - math.exp(-1 * c)) ** (j - 1)) * math.exp(-1 * c) * (c ** z) / math.factorial(z)

# This is the implicit function we solve in order to calculate the upper bound for 
# the optimal group-blind algorithm for a given level of bias
def func2(a, pq, k, j):
    return (1 - pq) * (1 - a ** k / math.factorial(k)) + pq * ((1 - math.exp(-1 * a)) ** (j - 1))

# We lower bound the designated slot here, where z is the number of slots that we 
# have designated.  
def lowerBoundC(p, q, k, z, j):
    T = math.factorial(k - z) ** (1 / (k - z))

    res = minimize(func1, 1, args=(z, j))
    C = res.x[0]
    
    success_T = math.exp(-1 * T) * sum(T ** i / math.factorial(i) for i in range(1, k)) 
    success_C = math.exp(-1 * C) * sum(C ** i / math.factorial(i) for i in range(1, z + 1))

    a = (1 - p) * success_T

    if k - z != 4:
        b = p * (1 - q) * success_T + \
            p * (1 - q) * (1 - math.exp(C - T) * (sum((T - C) ** i / \
            math.factorial(i) for i in range(0, k - z + 2))))
    else:
        b = p * (1 - q) * success_T + \
        p * (1 - q) * (1 - math.exp(C - T) * (sum((T - C) ** i / \
        math.factorial(i) for i in range(0, k - z + 3)))) * success_C


    c = p * q / j * ((1 - math.exp(-1 * T)) ** j) * success_T + \
        p * q * (1 - ((1 - math.exp(-1 * C)) ** (j - 1))) * success_C

    return a + b + c

# We upper bound the performance of a group-blind algorithm in biased environments.
def upperBoundT(p, q, k, j):
    res = fsolve(func2, math.factorial(k) ** (1 / k), args=(p * q, k, j), full_output=True)
    x = res[0][0]

    return (1 - p * q) * math.exp(-x) * sum(x ** i / math.factorial(i) for i in range(1, k + 1)) \
            + p * q / j * (1 - math.exp(-x)) ** j

