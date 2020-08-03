import gym
import numpy as np
import matplotlib.pyplot as plt
 

# parameters
intervalls = 20   
n_episodes_Q = 4000
n_episodes_lambQ = 1000
#epsilon = 0.1

def getState(observation):
    # converting input to integer to avoid floating point error
    xt = int(observation[0]*10)
    xt_p = int(observation[1]*100)
    state_pos = int(np.round((xt+12)*19/18))
    state_vel = int(np.round((xt_p+7)*19/14))
    state = state_pos * 20 + state_vel
    return state

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    init = True
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

def epsilonGreedy(Q,state,epsilon): # epsilon-greedy
    coin = np.random.rand()

    if coin < epsilon:
        action = np.random.randint(0,3)
    else:
        action = np.argmax(Q[state][:])
    return action

def QLearning(env):
    """
    performs training with basic q-learning

    Parameters
    ----------
    env : instance of gym cartpole environment

    Returns
    -------
    Q : Q-Function after training

    """
    
    Q = np.random.rand(400,3)
    e = np.ones([400,3])
    #state_dist = np.zeros([400,1])
    
    epsilon = 1 # use decaying value
    epsilon_min = 0
    gamma = 0.9
    alpha = 0.5
    lamb = 0.5
    render = False
    
    decay = 2 * epsilon/n_episodes_Q
    print("Decay = {}".format(decay))
    success = 0
    avg_reward_list = []
    reward_list = []
    for i in range(n_episodes_Q):
        
        # print episode and render 
        if i % 100 == 0:
            print("Episode {}".format(i))
            
            if i != 0:
                print("Goal reached in {} of 100 episodes. epsilon = {}".format(success,epsilon))
                success = 0

        if i % 1000 == 0 and i >3000:
            render = True
        else:
            render = False
        
        # init
        observation = env.reset()
        state = getState(observation)
        action = 1
        done = False
        step = 0
        tot_reward = 0
        #if epsilon > 0.1:
        #    epsilon = epsilon - 0.001
        while not done:
            #print(action)
            if render:
                env.render()
            observation, reward, done, info = env.step(action)
            #print("Step = {}".format(step))
            state_prime = getState(observation)
            a_prime = epsilonGreedy(Q,state_prime,epsilon)
            a_star = np.argmax(Q[state][:])
            delta = reward + gamma * Q[state_prime][a_star] - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * delta
            
            state = state_prime
            #state_dist[state]+=1
            action = a_prime
            step += 1
            tot_reward += reward

            if done and step <200:
                success +=1
       
        # calculate av rewards for last 100 episodes
        reward_list.append(tot_reward)
        if i % 100 == 0 and i > 1:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            
        if epsilon > epsilon_min:
                epsilon -= decay
                if epsilon < 0:
                    epsilon = 0
        render = False
    
    #print(avg_reward_list)
    plt.plot(100*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    return Q

def QLambda(env):
    
    # implementation of Watkins Q(lambda) algorithm
    Q = np.random.rand(400,3)
    e = np.ones([400,3])
    #state_dist = np.zeros([400,1])
    
    epsilon = 0.9 # use decaying value
    epsilon_min = 0.1
    gamma = 0.9
    alpha = 0.5
    lamb = 1
    render = False
    
    decay = 0.001
    print("Decay = {}".format(decay))
    success = 0
    avg_reward_list = []
    reward_list = []
    for i in range(n_episodes_lambQ):
        
        # print episode and render 
        if i % 20 == 0:
            print("Episode {}".format(i))
            
            if i != 0:
                print("Goal reached in {} of 20 episodes. epsilon = {}".format(success,epsilon))
                success = 0

        if i % 1000 == 0 and i >3000:
            render = True
        else:
            render = False
        
        # init
        observation = env.reset()
        state = getState(observation)
        action = 1
        done = False
        step = 0
        tot_reward = 0
        #if epsilon > 0.1:
        #    epsilon = epsilon - 0.001
        while not done:
            
            # render env if requested
            if render:
                env.render()
                
            observation, reward, done, info = env.step(action)
            state_prime = getState(observation)
            a_prime = epsilonGreedy(Q,state_prime,epsilon)
            a_star = np.argmax(Q[state_prime][:])
            delta = reward + gamma * Q[state_prime][a_star] - Q[state][action]
            
            e[state][action] = e[state][action] + 1
            
            # for all s,a
            for s in range(400):
                for a in range(3):
                    Q[state][action] = Q[state][action] + alpha * delta * e[state][action]
            if a_prime == a_star:
                e[state][action] = lamb*delta*e[state][action]
            else:
                e[state][action] = 0
                
            state = state_prime
            action = a_prime
            step += 1
            tot_reward += reward

            if done and step <200:
                success +=1
       
        # calculate av rewards for last 100 episodes
        reward_list.append(tot_reward)
        if i % 20 == 0 and i > 1:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            
        if epsilon > epsilon_min:
                epsilon -= decay
                if epsilon < 0:
                    epsilon = 0
        render = False
    
    #print(avg_reward_list)
    plt.plot(100*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    return Q

    



def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    #QLearning(env)
    QLambda(env)
    env.close()

if __name__ == "__main__":
    main()
