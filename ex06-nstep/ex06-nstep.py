import gym
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

def e_greedy(env, Q, s, epsilon):
    random = np.random.random_sample()
    if random <= epsilon:
        return np.random.choice(np.argwhere(Q[s,:]==np.max(Q[s,:])).reshape(-1))
    else:
        return np.random.randint(env.action_space.n)


def mc_Q_std(env, num_ep, epsilon=0.1):
    # initialize empty dictionaries of list
    returns = np.zeros((env.observation_space.n,  env.action_space.n))
    Q = np.zeros((env.observation_space.n,  env.action_space.n))  # state is key, value is value function
    # loop over episodes
    for i_episode in range(1, num_ep + 1):
        episode = []
        s = env.reset()
        done = False
        while not done:
            a = e_greedy(env, Q, s, epsilon)
            s_, r, done, _ = env.step(a)
            episode.append((s, a, r))
            s = s_

        N = np.zeros((env.observation_space.n,  env.action_space.n))
        states, actions, rewards = zip(*episode)

        # first-visit state-action value function
        for i, state in enumerate(states):
            N[state, actions[i]] += 1
            if N[state, actions[i]] == 1:
                returns[state, actions[i]] += np.sum(rewards[i:])
                Q[state, actions[i]] = np.mean(returns[state, actions[i]])
    return Q


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n,  env.action_space.n))

    for i in range(num_ep):
        S, A, R = [], [], [0]
        s = env.reset()
        S.append(s)
        a = e_greedy(env, Q, s, epsilon) #epsilon-greedy
        A.append(a)
        t = 0; T = np.inf
        while True:
            if t < T:
                s, r, done, _ = env.step(A[t])
                S.append(s)
                R.append(r)
                if done is True:
                    T = t + 1
                else:
                    a = e_greedy(env, Q, s, epsilon)
                    A.append(a)
            tao = t - n + 1
            if tao >= 0:
                G = 0
                for i in range(tao+1, min(tao+n, T) + 1):
                    G += (gamma ** (i-tao-1)) * R[i]
                Q[S[tao], A[tao]] += alpha * (G - Q[S[tao], A[tao]])
            if tao == T - 1:
                break 
            t += 1
    return Q


env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
Q_std = mc_Q_std(env, num_ep=10000)
RMS = np.zeros((10, 50))
N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
for n in range(10):
    a = 0
    for alpha in np.arange(0.01, 1.0, 0.02):
        Q_hat = nstep_sarsa(env, n=N[n], alpha=alpha, num_ep=1000)
        RMS[n,a] = np.sqrt(((Q_hat - Q_std) ** 2).mean())
        a += 1
        print(alpha)

x_axix = np.arange(0.01, 1.0, 0.02)
for n in range(len(N)):
    plt.plot(x_axix, RMS[n,:], color=np.random.rand(3,), label='n=' + str(N[n]))
plt.title('Result')
plt.xlabel("alpha")
plt.ylabel("RMS error on 10 episode")
plt.legend()
plt.show()
plt.savefig('ESTIMATIONQ' + '.png')
