import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break


def plot_V(Q, env, x, v):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    dims = Q.shape[:2]
    V = np.zeros(dims)
    for i in range(len(x)):
        for j in range(len(v)):
            V[i,j] = -np.min(Q[i,j,:])
        
    plt.imshow(V, origin='upper', 
               extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, 
               cmap=plt.cm.RdYlGn, interpolation='none')
    x_ = ['%.2f'%oi for oi in x]
    v_ = ['%.2f'%oi for oi in v]
    plt.xticks(np.arange(21), x_, rotation=75)
    plt.yticks(np.arange(21), v_)
    

def e_greedy(env, Q, s, epsilon):
    random = np.random.random_sample()
    if random <= epsilon:
        return np.random.choice(np.argwhere(Q[s[0], s[1],:]==np.max(Q[s[0], s[1],:])).reshape(-1))
    else:
        return np.random.randint(env.action_space.n)


def Q_lambda(env, alpha, gamma, lambda_, num_ep):
    x = np.arange(-0.6, 1.201, 0.09)
    v = np.arange(-0.07, 0.0701, 0.007)
    S = []
    for s in x:
        S_ = []
        for v_ in v:
            S_.append([s, v_])
        S.append(S_)
    S = np.array(S)
    Q = np.zeros((S.shape[0], S.shape[1], env.action_space.n))

    for episode in range(num_ep):
        #env.render()
        s = env.reset()
        index_x = (np.abs(x-s[0])).argmin()
        index_v = (np.abs(v-s[1])).argmin()
        z = np.zeros((S.shape[0], S.shape[1], env.action_space.n))
        a = e_greedy(env, Q, (index_x, index_v), 0.9)  # 0, 1, 2
        done = False
        while not done:
            s_, r, done, _ = env.step(a)
            index_x = (np.abs(x-s[0])).argmin()
            index_v = (np.abs(v-s[1])).argmin()
            index_x_ = (np.abs(x-s_[0])).argmin()
            index_v_ = (np.abs(v-s_[1])).argmin()

            a_ = e_greedy(env, Q, (index_x_, index_v_), 0.9)
            a_max = e_greedy(env, Q, (index_x_, index_v_), 1)
            delta = r + (gamma * Q[index_x_, index_v_, a_max]) - Q[index_x, index_v, a]
            z[index_x, index_v, a] += 1
            for i in range(len(x)):
                for j in range(len(v)):
                    for k in range(env.action_space.n):
                        Q[i,j,k] += alpha * delta * z[i,j,k]
                        if a_ == a_max:
                            z[i,j,k] *= gamma*lambda_
                        else:
                            z[i,j,k] = 0
            s = s_
            a = a_
        if episode % 20 == 0:
            plot_V(Q, env, x, v)
    return Q

def sarsa_lambda_LFA(env, alpha, gamma, lambda_, epsilon, num_ep):
    x = np.arange(-0.6, 1.201, 0.09)
    v = np.arange(-0.07, 0.0701, 0.007)
    S = []
    for s in x:
        S_ = []
        for v_ in v:
            S_.append([[s, v_, 1]]*3)
        S.append(S_)
    S = np.array(S)
    w = np.zeros(S.shape)
    Q = np.zeros((len(x), len(v), env.action_space.n))

    for episode in range(num_ep):
        #env.render()
        z = np.zeros(S.shape)
        s = env.reset()
        index_x = (np.abs(x-s[0])).argmin()
        index_v = (np.abs(v-s[1])).argmin()
        a = e_greedy(env, Q, (index_x, index_v), 0.9)  # 0, 1, 2
        F_a = S[index_x, index_v, a, :]
        done = False
        while not done:
            z[index_x, index_v, a, :] += 1  # accumulating traces
            s, r, done, _ = env.step(a)
            index_x = (np.abs(x-s[0])).argmin()
            index_v = (np.abs(v-s[1])).argmin()

            #a_ = e_greedy(env, Q, (index_x_, index_v_), 0.9)
            delta = r - np.sum(w[index_x, index_v, a, :])
            if not done:
                random = np.random.random_sample()
                if random <= epsilon:
                    for i in range(env.action_space.n):
                        F_a = S[index_x, index_v, i, :]
                        Q[index_x, index_v, i] = np.sum(w[index_x, index_v, i, :])
                    a = np.random.choice(np.argwhere(Q[index_x,index_v,:]==np.max(Q[index_x,index_v,:])).reshape(-1))
                else:
                    a = np.random.randint(env.action_space.n)
                    F_a = S[index_x, index_v, a, :]
                    Q[index_x, index_v, a] = np.sum(w[index_x, index_v, a, :])
                delta += gamma * Q[index_x, index_v, a]
            w += alpha * delta * z
            z *= gamma * lambda_
    return Q

def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    Q_lambda(env, 0.9, 0.9, 0.5, 1000)
    plt.show()
    #random_episode(env)
    env.close()


if __name__ == "__main__":
    main()
