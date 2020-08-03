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


def transition_probability_gen(env):
    transition_probability = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for tup in env.P[s][a]:
                transition_probability[s][a][tup[1]] += tup[0]
    return transition_probability


def expert_policy_generator(trajs, env):  # (a
    statistic = {}
    for traj in trajs:
        for state, action in traj:
            if state not in statistic:
                statistic[state] = [0] * env.action_space.n
            statistic[state][action] += 1
    for state in statistic:
        print(state, statistic[state])
        statistic[state] = np.array(statistic[state]) / sum(statistic[state])
    def expert(state):
        return np.random.choice(range(env.action_space.n), p=statistic[state])
    return expert


def find_feature_expectations(trajectories, env):  # (b
    feature_expectations = np.zeros(env.observation_space.n)

    for trajectory in trajectories:
        for state, _ in trajectory:
            feature_expectations += np.eye(env.observation_space.n)[state]

    feature_expectations /= len(trajectories)

    return feature_expectations


def find_expected_svf(env, trajectories, transition_probability, policy):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    n_trajectories = len(trajectories)
    trajectory_length = max([len(traj) for traj in trajectories])

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0][0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  (1 if j == policy[i] else 0) * # policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)


def irl(feature_expectations, trajectories, env, epochs, transition_probability):
    n_states = env.observation_space.n
    learning_rate = 0.005
    theta = np.random.uniform(size=(env.observation_space.n, ))
    for i in range(epochs):
        rewards = np.dot(np.eye(env.observation_space.n), theta)
        policy = value_iteration(env, rewards)
        expected_svf = find_expected_svf(env, trajectories, transition_probability, policy)
        grad = feature_expectations - np.array(range(env.observation_space.n)).T.dot(expected_svf)
        theta += learning_rate * grad
    return np.array(np.eye(env.observation_space.n)).dot(theta).reshape((n_states,))


def main():
    env = gym.make('FrozenLake-v0')
    transition_probability = transition_probability_gen(env)
    #env.render()
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    print("one trajectory is a list with (state, action) pairs:")
    print (trajs[0])
    print("")
    expert = expert_policy_generator(trajs, env)  # (a
    feature_expectations = find_feature_expectations(trajs, env)  # (b
    rewards = irl(feature_expectations, trajs, env, 100, transition_probability) # (c
    plot_rewards(rewards, env)


if __name__ == "__main__":
    main()
