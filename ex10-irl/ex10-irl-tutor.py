import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def generate_demonstrations(env, expertpolicy, epsilon=0.1, n_trajs=100):
    """ This is a helper function that generates trajectories using an expert policy """
    demonstrations = []
    for d in range(n_trajs):
        traj = []
        state = env.reset()
        for i in range(100):
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = expertpolicy[state]
            traj.append((state, action))  # one trajectory is a list with (state, action) pairs
            state, _, done, info = env.step(action)
            if done:
                traj.append((state, 0))
                break
        demonstrations.append(traj)
    return demonstrations  # return list of trajectories


def plot_rewards(rewards, env):
    """ This is a helper function to plot the reward function"""
    fig = plt.figure()
    dims = env.desc.shape
    plt.imshow(np.reshape(rewards, dims), origin='upper', 
               extent=[0,dims[0],0,dims[1]], 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(np.reshape(rewards, dims)[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def value_iteration(env, rewards):
    """ Computes a policy using value iteration given a list of rewards (one reward per state) """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)
    theta = 1e-8
    gamma = .9
    maxiter = 1000
    policy = np.zeros(n_states, dtype=np.int)
    for iter in range(maxiter):
        delta = 0.
        for s in range(n_states):
            v = V_states[s]
            v_actions = np.zeros(n_actions) # values for possible next actions
            for a in range(n_actions):  # compute values for possible next actions
                v_actions[a] = rewards[s]
                for tuple in env.P[s][a]:  # this implements the sum over s'
                    v_actions[a] += tuple[0]*gamma*V_states[tuple[1]]  # discounted value of next state
            policy[s] = np.argmax(v_actions)
            V_states[s] = np.max(v_actions)  # use the max
            delta = max(delta, abs(v-V_states[s]))

        if delta < theta: 
            break

    return policy




def main():
    env = gym.make('FrozenLake-v0')
    env.render()
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    print("one trajectory is a list with (state, action) pairs:")
    print (trajs[0])
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # task a)
    s_a_occucancies = np.zeros([env.observation_space.n,env.action_space.n])
    transitions = 0
    for demos in trajs:
        for pairs in demos:
            s_a_occucancies[pairs[0],pairs[1]] +=1
            transitions += 1
    print(s_a_occucancies)
    policy = np.argmax(s_a_occucancies,axis=1)
    print(policy)
    
    # task b)
    
    # calculate state visitation frequency
    # state_expec = np.zeros(env.observation_space.n)
    # features = np.zeros(env.observation_space.n)
    
    # for t in trajs:
    #     for s in t:
    #         state_expec[s[0]] += 1
    # state_expec = state_expec / len(trajs)
    
    # one-hot encoding of features
    features = np.identity(n_states)
    
               
    # calculate transition matrix p(s'|s,a)
    transition_matrix = np.zeros([n_states,n_actions,n_states]) # [s a s']
    for s in range(n_states):
        for a in range(n_actions):
            transitions = env.P[s][a]
            for p_trans,next_s,rew,done in transitions:
                transition_matrix[s,a,next_s] += p_trans
            transition_matrix[s,a,:]/=np.sum(transition_matrix[s,a,:])
    
    # calculate pi(a|s)
    policy_probs = np.zeros([n_states,n_actions])
    
    for s,a in enumerate(expertpolicy):
        policy_probs[s][a] = 1    
    
    
    # calculate mu_t
    mu_t = []
    mu_sum = np.zeros([n_states])
    mu = np.zeros([n_states])
    mu[0] = 1
    mu_t.append(mu)
    
    mu_sum += mu
    
    mu_t1 = np.zeros([n_states])
    # for s in range(n_states):
    #     mu_t1[s] = np.sum(transition_matrix[:,:,s] * mu.reshape(-1,1) * policy_probs)
        
    for i in range(100):
        mu = mu_t[i]
        mu_t1 = np.zeros([n_states])
        for s in range(n_states):
            mu_t1[s] = np.sum(transition_matrix[:,:,s] * mu.reshape(-1,1) * policy_probs)
        mu_t.append(mu_t1)
        mu_sum += mu_t1
        
    p_s_phi = mu_sum / 100
    print(p_s_phi)
    
    debug = True

if __name__ == "__main__":
    main()
