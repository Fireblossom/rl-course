import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """

    Parameters
    ----------
    state : numpy array
        contains state of cartpole environment.
    theta : numpy array
        contains parameters of linear features

    Returns
    -------
    numpy array
        return output of softmax function

    """
    z = state.dot(theta)
    exp = np.exp(z)
    return exp/np.sum(exp)



def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
            
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
     
    
    # policy parameters
    alpha = 0.025
    gamma = 0.99
    n_episodes = 800
    theta = np.random.rand(4, 2) 
    
    # init lists to store rewards of each episode and means of last 100 episodes 
    last_100_episodes = []
    episodes = []
    means = []
    
    for e in range(n_episodes):
        
        # render env every x steps
        if e % 100 == 0:
            states, rewards, actions = generate_episode(env, theta, True)
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        
        # keep track of previous 100 episode lengths
        if e < 100:
            last_100_episodes.append(sum(rewards))
        else:
            last_100_episodes.append(sum(rewards))
            last_100_episodes.pop(0)
        
        # compute mean
        mean = np.mean(last_100_episodes)
        means.append(mean)
        episodes.append(e)
        
        # learning rate decay
        if e % 200 == 0:
            # alpha = alpha/2
            if mean > 495:
                alpha = 0.00001 # slow down learning if mean of last 100 episodes is 500
            if mean < 495:
                alpha = 0.025
        # print mean every 100 episodes 
        if e % 100 == 0 or e == (n_episodes - 1):
            print("episode: " + str(e) + " Mean of last 100 episodes: " + str(mean)) 
        
        # REINFORCE Algorithm
        steps = len(states) # length of episode
        G_t = np.zeros([steps]) # init G_t
        for t in range(steps):
            # MC sampling of G_t
            for k in range(t+1,steps+1):
                G_t[t] += np.power(gamma,k-t-1) * rewards[k-1]
            pi = policy(states[t], theta)
            action = actions[t]
            # update rule
            theta[:,action] = theta[:,action] + alpha * np.power(gamma, t) * G_t[t] * (states[t] * (1 - pi[action]))
        
        # create plot
        plt.plot(episodes,means,'b')
        plt.xlabel("Episodes")
        plt.ylabel("Mean of last 100 episodes")
        plt.title("REINFORCE")

def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
