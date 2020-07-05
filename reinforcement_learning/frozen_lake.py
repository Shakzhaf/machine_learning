from __future__ import print_function
import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')


current_state = env.reset()
print(env.action_space)   #number of actions

print(env.observation_space)  #number of states

print("Number of actions : ",env.action_space.n)
print("Number of states : ",env.observation_space.n)

def epsilon_greedy(Q,current_state,na):
    epsilon = 0.3
    p = np.random.uniform(low=0,high=1)
    #print(p)
    if p > epsilon:
        return np.argmax(Q[current_state,:])#say here,initial policy = for each state consider the action having highest Q-value
    else:
        return env.action_space.sample()


# ## Q-Learning Implementation

#Initializing Q-table with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

#set hyperparameters
LEARNING_RATE = 0.5 #learning rate
DISCOUNT = 0.9 #discount factor lambda
EPISODES = 100000 #total episodes being 100000


for episode in range(EPISODES):
    # print('we are on episode: {}'.format(episode))
    current_state = env.reset()
    done = False
    while(True):
        action = epsilon_greedy(Q,current_state,env.action_space.n)
        new_state,reward,done,_ = env.step(action)
        if (reward==0):  
            if done==True:
                reward = -5 #to give negative rewards when holes turn up
                Q[new_state] = np.ones(env.action_space.n)*reward    #in terminal state Q value equals the reward
            else:
                reward = -1  #to give negative rewards to avoid long routes
        if (reward==1):
                reward = 100
                Q[new_state] = np.ones(env.action_space.n)*reward    #in terminal state Q value equals the reward
        Q[current_state,action] = Q[current_state,action] + LEARNING_RATE * (reward + DISCOUNT*np.max(Q[new_state,action]) - Q[current_state,action])
        current_state = new_state   
        if (done == True) :
                break

print("-------------------Q-table-------------------")
print(Q)


current_state = env.reset()
env.render()
while(True):
    action = np.argmax(Q[current_state])
    new_state,reward,done,_ = env.step(action)
    env.render()
    current_state = new_state
    if(done==True) :
        break
