import numpy as np
import matplotlib.pyplot as plt
import random

# set parameters
epsilon = 0.1
    
class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    # init variables (rewards, n_plays, Q) by playing each arm once
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    
    #play every arm once and calculate initial Q-values
    for arm in possible_arms:
        rewards[arm] = bandit.play_arm(arm)
        n_plays[arm]+=1
    
    Q = rewards/n_plays

    # Main loop
    while bandit.total_played < timesteps:
        
        x = np.argmax(Q)
        reward_for_a = bandit.play_arm(x)
        rewards[x]+=reward_for_a
        n_plays[x]+=1
        Q[x]=rewards[x]/n_plays[x]
        

def epsilon_greedy(bandit, timesteps):
    # init variables 
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    
    #play every arm once and calculate initial Q-values
    for arm in possible_arms:
        rewards[arm] = bandit.play_arm(arm)
        n_plays[arm]+=1
    
    Q = rewards/n_plays
    
    #main loop
    while bandit.total_played < timesteps:
        
        if np.random.uniform(0,1) <= epsilon:
            # exploration-step
            x = random.choice(possible_arms)
        else:
            # exploitation-step
            x = np.argmax(Q)
        
        reward_for_a = bandit.play_arm(x)
        rewards[x]+=reward_for_a
        n_plays[x]+=1
        Q[x]=rewards[x]/n_plays[x]
            


def main():
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.pdf')
    plt.show()


if __name__ == "__main__":
    main()
