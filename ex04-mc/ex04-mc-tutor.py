import gym
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


n_episodes = 500000
play_hit20 = False

def getValueFunction(rewardList):
    stateValueFunction = np.zeros((2,10,10))
    for x in range(0,2):
        for y in range(0,10):
            for z in range(0,10):
                rewardSum = sum(rewardList[x][y][z][0]) +sum(rewardList[x][y][z][1])
                n_elements = len(rewardList[x][y][z][0]) + len(rewardList[x][y][z][1])
                if n_elements > 0:
                    stateValueFunction[x][y][z] = rewardSum/n_elements
                else:
                    stateValueFunction[x][y][z] = 0
                
    return stateValueFunction
                    

def main(episodes, policy="optimal"):
    env = gym.make('Blackjack-v0')
    
    # start monte carlo simulation
     
    returnOfStates = [] # create a 2x10x10 dimensional list which stores the returns for every state
    for x in range(0,2):
        returnOfStates.append([])
        for y in range(0,10):
            returnOfStates[x].append([])
            for z in range(0,10): 
                returnOfStates[x][y].append([])
                for a in range(0,2):
                    returnOfStates[x][y][z].append([])
    
    qFunction = np.zeros([2,10,10,2])
    strategy = np.random.choice([0,1],size=(2,10,10))
    
    episodeCounter = 0
    startTime = time.time()
    while episodeCounter < episodes:
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        
        # getting indices to access state-value function
        index0 = int(obs[2]) # 0 or 1 for usable ace
        index1 = obs[0]-12   # value of player card
        index2 = obs[1]-1    # value of dealer card
        stateList =[]
        while not done:
            
            if ((policy == "hit20") or (obs[0]<12)):
                if obs[0] >= 20:
                    #print("stick")
                    action = 0
                    obs, reward, done, _ = env.step(action)
                else:
                    #print("hit")
                    action = 1
                    obs, reward, done, _ = env.step(action)
            else:
                action = strategy[index0][index1][index2]
                obs, reward, done, _ = env.step(action)
                
            if index1 >= 0: # skip states where player_sum is below 1
                returnOfStates[index0][index1][index2][action].append(reward)
                #if episodeCounter:
                #    qFunction[index0][index1][index2][action] = sum(returnOfStates[index0][index1][index2][action])/len(returnOfStates[index0][index1][index2][action])
                #    strategy[index0][index1][index2] = np.argmax(qFunction[index0][index1][index2])            
                stateList.append((index0,index1,index2,action))
            index0 = int(obs[2]) # update state
            index1 = obs[0]-12
            index2 = obs[1]-1
        
        for state in stateList:
            index0 = state[0]
            index1 = state[1]
            index2 = state[2]
            action = state[3]
            #returnOfStates[index0][index1][index2][action].append(reward)
            qFunction[index0][index1][index2][action] = sum(returnOfStates[index0][index1][index2][action])/len(returnOfStates[index0][index1][index2][action])
            strategy[index0][index1][index2] = np.argmax(qFunction[index0][index1][index2])
            
        if (episodeCounter % 50000 == 0):
            print("Episode: {}".format(episodeCounter)) # visualize simulation progress
        episodeCounter+=1
    
    stopTime = time.time()
    simTime = stopTime - startTime
    valueFunction = getValueFunction(returnOfStates) # calculate value function from return list
    print("Simulation of {} episodes took {} seconds".format(n_episodes,simTime))
    print(strategy)
    
    # plotting results 
    
    fig = plt.figure(figsize=plt.figaspect(1.2),dpi=200)
    x = np.arange(1,11,1)
    y = np.arange(12,22,1)
    x,y = np.meshgrid(x,y)
    
    ax = fig.add_subplot(2,1,1,projection='3d') #subplot for no usable ace
    aceFalse = valueFunction[0]
    ax.plot_wireframe(x,y,aceFalse)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('no usable ace')
    
    ax = fig.add_subplot(2,1,2,projection='3d') #subplot for usable ace
    aceTrue = valueFunction[1]
    ax.plot_wireframe(x,y,aceTrue)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('usable ace')
    plt.savefig("prediction_figure.pdf")
    
    
if __name__ == "__main__":
    if play_hit20:
        main(n_episodes, policy="hit20")
    else:
        main(n_episodes, policy="optimal")
