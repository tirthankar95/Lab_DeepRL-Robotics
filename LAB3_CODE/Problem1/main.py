from collections import defaultdict
import numpy as np
import gym
import gym_examples
import math
import matplotlib.pyplot as plt
    
def qlearning(g_,t_,w_,d_,p_):
    def printRL():
        '''
        0 -> Right
        1 -> Down
        2 -> Left
        3 -> Up
        '''
        aN=env.action_space.n
        direction={0:'R',3:'U',2:'L',1:'D'}
        policy=[['X' for i in range(aN)] for j in range(aN)]
        qValue=[[0 for i in range(aN)] for j in range(aN)]
        sValue=[[0 for i in range(aN)] for j in range(aN)]
        for key,value in Q.items():
            #row & cols are reversed.
            qValue[key[1]][key[0]]=value
            sValue[key[1]][key[0]]=max(value)
            policy[key[1]][key[0]]=direction[np.argmax(value)]
        for _ in policy:
            print(_)
        for _ in sValue:
            print([round(__,2) for __ in _])
        print()
    env=gym.make('gym_examples/GridWorldTm-v0',size=4,g=g_,t=t_,w=w_,p=p_)
    Q=defaultdict(lambda: np.zeros(env.action_space.n))
    a_Orig=0.8 # learning rate.
    epsilonOrig=1 # discovery. 
    episodes=50000
    aN=env.action_space.n
    x=[]
    y=[]
    for _ in range(1,episodes+1,1):
        env.action_space.seed(2000)
        obs,info=env.reset(seed=2022)
        s=tuple(obs['agent'])
        if _%episodes==0:
            print('Ep_no {}; Epsilon {}; Lr {}'.format(_,epsilon,a_))
            printRL()
        rewardR=0
        epsilon=epsilonOrig*math.exp(-(10*_/episodes))
        a_=a_Orig*math.exp(-(100*_/episodes))
        while True:
            def policy(s):
                actionProb=np.ones(aN)*(epsilon/(aN))
                actionProb[np.argmax(Q[s])]+=(1-epsilon)
                return actionProb
            actionProb=policy(s)
            action_i=np.random.choice(np.arange(aN),p=actionProb)
            obsN,reward,terminated,truncated,info=env.step(action_i)
            rewardR+=reward
            ns=tuple(obsN['agent'])
            if terminated:
                Q[s][action_i]+=a_*(reward-Q[s][action_i])
                break
            next_best_action=np.argmax(Q[ns])
            Q[s][action_i]+=a_*(reward+d_*Q[ns][next_best_action]-Q[s][action_i])
            s=ns
        if _%10==0:
            y.append(rewardR)
            x.append(_)
    env.close()
'''
    plt.plot(x,y)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Reward as a funciton of Episodes.')
    plt.show()
'''
    #print(y)

if __name__=='__main__':
    runs=2
    for run in range(runs):
        print("Run{} ".format(run))
        qlearning(50,-50,-1,0.9,0.7)
        qlearning(100,-50,-5,0.95,0.1)
        qlearning(50,-100000,-20,0.9,0.7)
        print('---------------------------')
