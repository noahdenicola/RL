#Noah De Nicola
#08/22
#Lunar Lander DQN

import torch
import numpy as np
import gym
import time, math, random
import pyglet
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from torch.distributions import Categorical

import wandb
import pprint

#Network CLass
class Network(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()

        #Layers
        self.fc1 = nn.Linear(8, hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.fc3 = nn.Linear(hidden[1],4)

        #Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x

#Agent Class
class DQN_Agent():
    def __init__(self, hidden, lr, discount, K, dropout, lossFunc):

        #Networks
        self.main = Network(hidden, dropout)
        self.target = self.main
        self.target.dropout = nn.Dropout(0.0)

        #Hyper parameters
        self.lr = lr
        self.discount = discount
        self.A = torch.rand([K, 8])
        self.CMS = CountMinSketch()
        
        #Criterion
        if lossFunc == 'MSE':
            self.criterion = nn.MSELoss()
        elif lossFunc == 'Huber':
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.SmoothL1Loss()

        #Optimizer     
        self.optimizer = optim.Adam(self.main.parameters(), lr=lr)

    #Save model
    def save(self, filename):
        torch.save(self.main.state_dict(), "saved/"+filename+".pth")

    #Load model
    def load(self, filename):
        sd = torch.load("saved/"+filename+".pth")
        self.main.load_state_dict(sd)
    
    #Sync Parameters
    def equate(self):
        self.target.load_state_dict(self.main.state_dict())

    #Set main/target to evalutaion mode(dropout off)
    def eval(self):
        self.main.eval()
        self.target.eval()

    #Set main/target to training mode(dropout on)
    def train(self):
        self.main.train()
        self.target.train()

    #Forward pass of network, defaulted to target
    def forward(self, state, target=False):
        if target:
            return self.target(state)
        else:
            return self.main(state)

    #Returns and increments hash count
    def count(self, state):
        hashCode = self.hash(state)
        return self.CMS.count(hashCode)

    #Hash function hashes transitions (not states).
    def hash(self, state):
        state = torch.FloatTensor(state)
        #can preProcess
        B = torch.matmul(self.A, state)
        S = torch.sign(B)
        ssum = 0
        po = 0
        for s in S:
            i = (abs(s.item())+s.item())/2
            ssum += i*(2**po)
            po+=1

        return int(ssum)

#Count Min Sketch (Hashing) Class
class CountMinSketch():
    def __init__(self):
        self.stateCount = 0
        self.primes = [999931, 999953, 999959, 999961, 999979, 999983]
        self.hashTable = []
        for i in self.primes:
            table = np.zeros(i, dtype=np.int32)
            self.hashTable.append(table)

    def count(self, hashCode):
        mini = np.infty
        for i in range(len(self.primes)):
            modHash = hashCode%self.primes[i]
            self.hashTable[i][modHash] += 1
            count = self.hashTable[i][modHash]
            if count < mini:
                mini = count

        if mini == 1:
            self.stateCount += 1

        return mini

#Bellman equation to find training labels
#Memory = [state, action, reward, nextState, done]
def bellman(agent, memory, indicies, exploreBonus):
    #Initialise
    q_target = torch.zeros([len(indicies)])
    q_expected = torch.zeros([len(indicies)])
    loc = 0
    for i in indicies:
        c = agent.count(memory[i][0]) #get state visits
        r = exploreBonus/(math.sqrt(1+c)) #get bonus reward [set to 0 to stop CBE-H]
        rr = memory[i][2]+r #compute total reward
        nxt = torch.max(agent.forward(memory[i][3], target=True)) #get max q value of next state
        if memory[i][4]: #unless terminal
            nxt = 0
        q_target[loc] = (rr+agent.discount*nxt) #target
        q_hat = agent.forward(memory[i][0], target=False)
        q_expected[loc] = q_hat[memory[i][1]] #expected
        loc +=1
    return q_expected, q_target

#Automated training loop, integrated with wandb
def auto_train(config=None):
    #torch.manual_seed(13523531624120967073)

    #wandb
    with wandb.init(config=config):
        config = wandb.config

        #Enviroment
        env = gym.make(config.env, new_step_api=True)

        #Agent
        pi = DQN_Agent( [config.hidden1, config.hidden2], config.lr, config.discount, config.K, config.dropout, config.lossFunc)

        #Training
        scoreQ = deque(maxlen=100)
        memory = deque(maxlen=config.bufferSize)
        best = -np.infty #best score
        e = -1
        steps = 0
        transitions = 0

        #greed
        eps = 1 #epsilon
        decay = config.decay

        #Initialise 
        aveScore = 0
        solved = False

        #Fixed training steps (not epochs)
        while steps < config.totalSteps:
            e +=1

            #Initialise
            state = env.reset() #state
            score = 0 #epoch score

            #Epoch Loop
            for t in range(config.playLength):

                #Experience 
                with torch.no_grad():
                    pi.eval()
                    
                    #Get Q-value of each action in state
                    q_val = pi.forward(state, target=False).detach().numpy()

                    #For \epsilon-greedy
                    if random.random() < eps:
                        action = random.randint(0,1)
                    else:
                        action = np.argmax(q_val) #greedy

                    #Transition
                    nextState, reward, done, other, _ = env.step(action)
                    transitions+=1

                    #Sum reward
                    score += reward
                    
                    #Store experience 
                    memory.append([state, action, reward, nextState, done])
                    
                    #move into new state
                    state = nextState
                
                #Learning
                if transitions%config.update_every==0 and t>0:

                    #Get batched data
                    l = len(memory)
                    indicies = random.sample(range(l), min(config.batchSize, l))

                    pi.train()

                    #Get labels
                    q_expected, q_target = bellman(pi, memory, indicies, config.bonus)
                        
                    #Update Network
                    pi.optimizer.zero_grad()
                    loss = pi.criterion(q_expected, q_target)
                    loss.backward()
                    pi.optimizer.step()
                    steps +=1
                                        
                    wandb.log({"loss": loss}) #log loss to wandb

                    #Sync Target and Main networks (target <-- main) 
                    if steps%config.C == 0:
                        pi.equate() 
               
                #end_t = t

                #Terminal State 
                if done or other:
                    break
            
            #Log epoch score, total unique states, epoch and steps to wandb
            wandb.log({"score": score, "stateCount": pi.CMS.stateCount, "epoch": e, "gradient_steps": steps})
            
            #Evaluate Agent
            if not solved:
                if e%100 == 0 or score>=200: #Every 100 epochs or when a perfect score is reached
                    total = 0
                    for p in range(100):
                        tempEnv = gym.make(config.env, new_step_api=True)
                        total += sim(pi, tempEnv)
                        tempEnv.close()

                    aveScore = total/100
            
            #\epsilon-greedy
            if decay != 0:
                eps = max(0.01, eps*decay)
            else:
                eps = 0

            #Save network parameters if best score
            if aveScore > best:# and e >= 1:
                best = aveScore
                pi.save('LL_best')
                wandb.log({"bestScore": aveScore})
                print("%8d steps\t Episode %5d\t Accuracy %6.2f\t (%d)"% (steps, e, score, pi.CMS.stateCount))

                
            if aveScore >= 200 and not solved:
                print("")
                print('Solved in: {} steps\t {} episodes\t {} states visited'.format(steps, e, pi.CMS.stateCount))
                wandb.log({"solved_in": steps})
                solved = True
                break
                
            if e%100==0 and e>1:
                simEnv = gym.make(config.env, render_mode = 'human', new_step_api=True)
                simAgent = pi
                simAgent.load('LL_best')
                sim(simAgent, simEnv)
                simEnv.close()

            #Epoch tracing
            if not solved:
                print("%8d steps\t Episode %5d\t Accuracy %6.2f\t (%d)"% (steps, e, score, pi.CMS.stateCount), end="\r")

        env.close()

    


#Lander agent simulation 
def sim(agent, env):
    agent.eval()
    done = False
    state = env.reset()
    reward = 0
    s = 0
    while(not done and s < 1200 ):
        s += 1
        state, r, done, i, _ = env.step(torch.argmax(agent.forward(state, target=False)).item())
        reward +=r

    return reward

#Simulate and Render
def showOff(agent):
    x = ""
    while(x!="q"):
        env = gym.make("LunarLander-v2", render_mode = 'human', new_step_api=True)
        env.reset()
        print(sim(agent, env))
        env.close()
        x = input("Press enter to restart sim or q to quit: ")

#main
def main():
    #Weights and Bias (wandb)
    wandb.login()
    sweep_config = {'method':'grid'}
    metric = {  'name':'solved_in', 'goal':'minimize'}
    sweep_config['metric'] = metric
    parameters_dict = { "env":          {'value' : "LunarLander-v2"},
                        "playLength":   {'value' : 100 }, 
                        "lossFunc":     {'value' : 'Huber'},
                        'lr' :          {'value' : 0.001 },
                        'K':            {'value' : 128 },
                        'update_every': {'value' : 25},
                        'C' :           {'value' : 30},
                        "totalSteps":   {'value' : 60000}, 
                        'bufferSize':   {'value' : 10000 }, 
                        "discount":     {'value' : 0.99},
                        "bonus":        {'value' : 1},
                        "dropout":      {'value' : 0.5},
                        'batchSize':    {'value' : 32},
                        'decay' :       {'value' : 0 },
                        'hidden1' :     {'value' : 64 },
                        'hidden2' :     {'value' : 64 }}
    sweep_config['parameters'] = parameters_dict
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="LunarLander_CMS-DQN")
    wandb.agent(sweep_id, auto_train) 
    wandb.finish()

if __name__ == "__main__":
    main()
