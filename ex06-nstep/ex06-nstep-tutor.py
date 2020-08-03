import gym
import numpy as np
import matplotlib.pyplot as plt

import pdb
import sys
import traceback


def epsilonGreedy(epsilon,state,Q):
    if state == 64:
        debug = True
        return 0
    if np.random.rand() < epsilon:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q[state][:])
    return action

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    
    Q = np.zeros([64,4]) # 16 states and 4 actions
    
    for i in range(num_ep):
        if (i % 1000) == 0:
            print("Episode: {}".format(i))
        s = []
        a = []
        r = []
        s_ = env.reset() # init s0
        s.append(s_)
        done = False
        a_ = epsilonGreedy(epsilon,s[0],Q)
        a.append(a_)
        T = np.inf
        t = 0
        while True:
            if t < T:
                s_, r_, done, _ = env.step(a[t])
                s.append(s_)
                r.append(r_)
                if done:
                    T = t + 1
                else:
            
                    a_ = epsilonGreedy(epsilon,s[t+1],Q)
                    
                    a.append(a_)
                tau = t - n +1
                    
            if tau >= 0:
                if T == np.inf:
                    upper_index = tau + n
                else:
                    upper_index = min(tau + n, T)
                lower_index = tau + 1
                G = 0
                for i in range(upper_index, lower_index +1):
                    G += np.power(gamma,i-tau-1)*r[i-1]
                if (tau + n) < T:
                    G = G + np.power(gamma,n) * Q[s[tau + n]][a[tau + n]]
                    Q[s[tau]][a[tau]] = Q[s[tau]][a[tau]] + alpha * (G - Q[s[tau]][a[tau]])
            t+=1
            if tau == (T - 1): break
                        
                    
                    
            
    
    


env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
# try:
#     nstep_sarsa(env)
# except:
#     type, value, tb = sys.exc_info()
#     traceback.print_exc()
#     pdb.post_mortem(tb)
nstep_sarsa(env)