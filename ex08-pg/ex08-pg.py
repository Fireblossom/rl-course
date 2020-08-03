import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    pi_0 = float(1 / (1+np.exp(-np.dot(state, theta))))
    #print(pi_0)
    return [pi_0, 1-pi_0]  # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = [[]]
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        #print(p)
        action = np.random.choice(len(p), p=p)
        actions.append(action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
        states.append(state)
        
    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 1)*2-1  # policy parameters
    gamma = 1
    alpha = 0.0001

    length = []
    aves = []
    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, False)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)
        length.append(len(states))
        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean
        if e > 100:
            ave = np.mean(length[-100:])
            aves.append(ave)
        else:
            ave = 0
        if ave >= 495:
            break
        # TODO: implement the reinforce algorithm to improve the policy weights
        for t in range(len(states)):
            G = 0
            for k in range(t+1, len(states)+1):
                G += (gamma ** (k-t-1)) * rewards[k]
            theta += alpha * (gamma ** t) * G * np.dot(np.array(states).T, (1/(1+np.exp(-np.dot(states, theta))))-1)
            #print(theta)
    return aves


def main():
    env = gym.make('CartPole-v1')
    aves = REINFORCE(env)
    plt.plot(range(100, 10000-1),aves)
    plt.xlabel('Epochs')
    plt.ylabel('Average length')
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
