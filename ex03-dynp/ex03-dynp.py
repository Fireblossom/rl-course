import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    delta = 1
    count = 0
    while delta > theta:
        count += 1
        delta = .0
        for s in range(n_states):
            v = tuple(V_states) # deep copy
            sum_up = 0
            for a in range(n_actions):
                args = env.P[s][a]
                sum_up = 0
                for arg in args:
                    p, n_state, r, is_terminal = arg
                    sum_up += p * (r + gamma * V_states[n_state])
                V_states[s] = max(sum_up, V_states[s])
            delta = max(delta, np.linalg.norm(V_states - v))

    policy = np.zeros(n_states)
    for s in range(n_states):
        sum_up_action = []
        for a in range(n_actions):
            args = env.P[s][a]
            sum_up = 0
            for arg in args:
                p, n_state, r, is_terminal = arg
                sum_up += p * (r + gamma * V_states[n_state])
            sum_up_action.append(sum_up)
        policy[s] = np.argmax(sum_up_action)

    return policy, V_states, count
    

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy, value_optimal, steps = value_iteration()
    print("Computed policy:")
    print(policy)
    print("Optimal value:")
    print(value_optimal)
    print("Steps to converge:", steps)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
