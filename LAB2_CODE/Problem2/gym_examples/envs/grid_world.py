from collections import defaultdict
import numpy as np
import gym
import gym_examples


def printRL():
    '''
    0 -> Right
    1 -> Down
    2 -> Left
    3 -> Up
    '''
    direction={0:'R',3:'U',2:'L',1:'D'}
    policy=[['X' for i in range(aN)] for j in range(aN)]
    qValue=[[0 for i in range(aN)] for j in range(aN)]
    sValue=[[0 for i in range(aN)] for j in range(aN)]
    for key,value in Q.items():
        #row & cols are reversed.
        qValue[key[1]][key[0]]=value
        sValue[key[1]][key[0]]=max(value)
        policy[key[1]][key[0]]=direction[np.argmax(value)]
    print(sValue)
    print(policy)
    
env=gym.make('gym_examples/GridWorldTm-v0',size=4,g=50,t=-50,w=-1,p=0.7)
Q=defaultdict(lambda: np.zeros(env.action_space.n))
a_=0.75 # learning rate.
d_=0.9 # discount factor. 
epsilon=0.1 # discovery. 
episodes=1000
aN=env.action_space.n 

for _ in range(1,episodes+1,1):
    env.action_space.seed(2000)
    obs,info=env.reset(seed=2022)
    s=tuple(obs['agent'])
    if _%100==0:
        print('NO {}'.format(_))
        printRL()
    while True:
        def policy(s):
            actionProb=np.ones(aN)*(epsilon/(aN))
            actionProb[np.argmax(Q[s])]+=(1-epsilon)
            return actionProb
        actionProb=policy(s)
        action_i=np.random.choice(np.arange(aN),p=actionProb)
        obsN,reward,terminated,truncated,info=env.step(action_i)
        ns=tuple(obsN['agent'])
        if terminated:
            Q[s][action_i]+=a_*(reward-Q[s][action_i])
            break
        next_best_action=np.argmax(Q[ns])
        Q[s][action_i]+=a_*(reward+d_*Q[ns][next_best_action]-Q[s][action_i])
        s=ns
env.close()
