#Noah De Nicola
#07/22
#CartPole Reinforce


import torch
import numpy as np
import gym
import time, math, random
import pyglet
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from collections import deque

class Policy(nn.Module): 
    def __init__(self, hidden=20, dropout=0):
        super().__init__()

        #Layers
        self.input = nn.Linear(4,hidden)
        self.out = nn.Linear(hidden,2)

        #Parameters
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = F.relu(self.dropout(self.input(x)))
        x = F.softmax(self.out(x), dim=1)

        return x

    def act(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
def save(pi):
    torch.save(pi.state_dict(), "saved/CP_RNFRC_best.pth")

def load(pi):
    sd = torch.load("saved/CP_RNFRC_best.pth")
    pi.load_state_dict(sd)
    return pi

def train(epochs, gamma, lr, T):
    #Initialise Environment, agent and optimizer 
    cp = gym.make('CartPole-v1', new_step_api=True)
    pi = Policy()
    optimizer = optim.Adam(pi.parameters(), lr=lr)
    
    scores = []
    sd = deque(maxlen=50)
    best = 0
    for i in range(epochs):
        #Initialise epoch
        log_probs = []
        rewards = []
        s = cp.reset()
        for t in range(T):
            a, logp = pi.act(s)
            log_probs.append(logp)
            s, r, done, info, _ = cp.step(a)
            rewards.append(r)
            if done:
                break
        score = sum(rewards)
        scores.append(score)
        sd.append(score)
        aveScore = np.mean(sd)

        #Compute Future Return
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        p_loss = []

        for lp in log_probs:
            p_loss.append(-lp*R)
            
        p_loss = torch.cat(p_loss).sum()

        optimizer.zero_grad()
        p_loss.backward()
        optimizer.step()

        if aveScore >= best:
            best = aveScore
            save(pi)
            print('Episode {}\t Accuracy: {:.2f}'.format(i, (aveScore/500)*100))
            
        if aveScore>=500.0:
            print('Environment solved in {:d} episodes!'.format(i))
            break
        
        if i%250 == 0:
            temp_pi = Policy()
            temp_pi = load(temp_pi)  
            tcp = gym.make('CartPole-v1', new_step_api=True, render_mode='human')
            sim(temp_pi, tcp, True)

        print('Episode {}\t Accuracy: {:.2f}'.format(i, (score/500)*100), end="\r") 

    load(pi)
    return pi

def showOff(pi):
    x = ""
    while(x!="q"):
        cp = gym.make("CartPole-v1", new_step_api=True, render_mode='human')
        obs = cp.reset()
        print(sim(pi, cp))
        cp.close()
        x = input("Press enter to restart sim or q to quit: ")

def sim(pi, cartpole, render=True):
    done = False
    s = cartpole.reset()
    count = 0
    while(not done and count<500):
        count +=1
        a, l = pi.act(s)
        s, r, done, i, _ = cartpole.step(a)
    cartpole.close()
    return count

def main():
    opi = train(epochs=5000, gamma=0.99, lr=0.01, T=500) 
    showOff(opi)
    

if __name__ == '__main__':
    main()
