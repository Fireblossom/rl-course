import gym
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    R = {
        True: [],
        False: []
    }
    for d in range(10): # (12~21)
        R[True].append(np.empty((10, 0)).tolist())
        R[False].append(np.empty((10, 0)).tolist())
    V = {
        True: [], 
        False: []
    }

    # generate episodes
    for __ in range(10000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        episode = []
        done = False
        while not done:
            # print("observation:", obs)
            episode.append(obs)
            if obs[0] >= 20:
                # print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                # print("hit")
                obs, reward, done, _ = env.step(1)
            # print("reward:", reward)
            # print("")
        
        for o in episode:
            if o[0] >= 12:
                R[o[2]][o[1]-1][o[0]-12].append(reward)
    
    for d in range(10):
        V[True].append(list([np.mean(i) for i in R[True][d]]))
        V[False].append(list([np.mean(i) for i in R[False][d]]))

    
    print("V for first-visit MC:")
    print(V)
    plot_blackjack_values(V)


def plot_blackjack_values(V, filename = 'first-visit_10000_v'):
    def get_Z(x, y, usable_ace):
        return V[usable_ace][x-12][y-1]

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.savefig(filename+'.svg')
    plt.savefig(filename + '.png')
    plt.show()


def mc_es():
    env = gym.make('Blackjack-v0')
    q = {
        True: np.zeros((10,2)),
        False: np.zeros((10,2))
    }
    pi = {
        True: np.zeros(10, dtype=np.intc),
        False: np.zeros(10, dtype=np.intc)
    }
    R = {
        True: [],
        False: []
    }
    for r in range(10):
        R[True].append(np.empty((2, 0)).tolist())
        R[False].append(np.empty((2, 0)).tolist())

    for __ in range(10000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        a0 = np.random.randint(2)
        episode = [(obs[0], a0, obs[2])]
        obs, reward, done, _ = env.step(a0)
        while not done:
            episode.append((obs[0], pi[obs[2]][obs[0]-12], obs[2]))
            obs, reward, done, _ = env.step(pi[obs[2]][obs[0]-12])
        
        for o in episode:
            if o[0] >= 12:
                R[o[2]][o[0] - 12][o[1]].append(reward)
                q[o[2]][o[0] - 12][o[1]] = np.mean(R[o[2]][o[0] - 12][o[1]])

        for s in range(10):
            pi[True][s] = np.argmax(q[True][s])
            pi[False][s] = np.argmax(q[False][s])

    print("With usable ace:")
    print(pi[True])
    print("Without usable ace:")
    print(pi[False])


if __name__ == "__main__":
    main()
    mc_es()
