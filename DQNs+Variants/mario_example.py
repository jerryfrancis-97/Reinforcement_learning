import gym_super_mario_bros as mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from wrappers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

class Duelling_Double_DQN_ImageGames(nn.Module):
    def __init__ (self, lr,n_frames,n_actions,name,checkpt_dir="./mario_models"):
        super(Duelling_Double_DQN_ImageGames,self).__init__()
        self.lr = lr
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'.pth')

        #layers
        self.layer1 = nn.Conv2d(n_frames, 32, 8, 4)
        self.layer2 = nn.Conv2d(32,64,4,2)#(32,64, 3,1)
        self.layer3 = nn.Conv2d(64,64,3,stride=1)
        self.dense1 = nn.Linear(3136,512) #(20736,512)
        self.A = nn.Linear(512, n_action)
        self.V = nn.Linear(512, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')#///
        self.to(self.device)
    
    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1,3136)#(-1,20736)
        x = F.relu(self.dense1(x))
        A = self.A(x)
        V = self.V(x)

        # q = v + (adv - 1/adv.shape[-1] * adv.max(-1,True)[0])
        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(-1,20736)
        x = F.relu(self.dense1(x))
        A = self.A(x)

        return A

    def save_checkpoint(self):
        print("SAving checkpoint....")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint....")
        self.load_state_dict(torch.load(self.checkpoint_file))

    
class Replay_memory(object):
    def __init__(self,N):
        self.memory = deque(maxlen=N)
        self.mem_counter = 0

    def push(self,transition):
        self.memory.append(transition)
        self.mem_counter+=1

    def sample(self,n):
        return random.sample(self.memory,n)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, exp_name, n_frames=4, epsilon_dec=1e-3,
                    epsilon_min=0.01, mem_size=1000000, replace=1000):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.replace = replace
        self.batch_size = batch_size
        self.lr = lr
        
        self.learn_step_counter = 0
        self.memory = Replay_memory(mem_size)
        self.q_eval = Duelling_Double_DQN_ImageGames( self.lr,n_frames,n_actions,'q_eval_'+exp_name)
        self.q_next = Duelling_Double_DQN_ImageGames( self.lr,n_frames,n_actions,'q_next_'+exp_name)

        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        
        return action
    
    def learn(self):
        # ce = nn.MSELoss()
        if self.memory.mem_counter< self.batch_size:
            return
        
        if self.learn_step_counter % self.replace == 0:
            with torch.no_grad():
                # for params_next, params_eval in zip(self.q_next.parameters(), self.q_eval.parameters()):
                #     params_next.data = params_eval.data
                weights = self.q_eval.state_dict()
                self.q_next.load_state_dict(weights)
        
        s,r,a,next_state,done = map(np.array, zip(*self.memory.sample(batch_size)))
        
        s = s.squeeze()
        next_state = next_state.squeeze()
        a_max = self.q_eval(next_state).max(1)[1].unsqueeze(-1)
        r = torch.FloatTensor(r).unsqueeze(-1).to(device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            y = r + self.gamma * self.q_next(next_state).gather(1,a_max)*int(1-done)
        a = torch.tensor(a).unsqueeze(-1).to(device)
        qq = torch.gather(self.q_eval(s), dim=1, index=a.view(-1, 1).long())

        loss = F.smooth_l1_loss(qq,y).mean()

        q_eval.optimizer.zero_grad()
        loss.backward()
        q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter+=1

        return loss


def arange(s):
    if not type(s) == 'numpy.ndarray':
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret,0)


env = mario.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = wrap_mario(env)
agent = Agent(lr=0.01, gamma=0.99, n_actions=env.action_space.n, epsilon=1.0, batch_size=64, exp_name='SuperMarioBros-v0', n_frames=4)
games=1000
per_episode_score=[]
cumulative_score=[]

for i in range(games):
    s = arange(env.reset())
    done=False

    while not done:

        action = agent.choose_action(s)
        next_state, reward, done, _ = env.step(action)
        
        print("old : ",next_state.shape)
        next_state = arange(next_state)
        print("new : ",next_state.shape)

        episode_score += r
        

        #reward clipping
        r = np.sign(r)* (np.sqrt(abs(r)+1)-1) + 0.001 * r
        print(r,' <------the reward')
        agent.store_transition(s, float(r), int(a), next_state, int(1-done))
        s = next_state
        agent.learn()
        stage = env.unwrapped._stage
        print("Timestep done------")
    
    cum_reward += episode_score
    if i%100==0:
        print("%s |Epoch : %d | score : %f | cum_reward: %f | loss : %.2f | stage : %d" % (device, i, episode_score/1000, cum_reward, loss/1000, stage))
        per_episode_score.append(episode_score / 100)
        cumulative_score.append(cum_reward)
        episode_score = 0
        loss = 0.0
        pickle.dump(per_episode_score, open('per_epi_score.p', 'wb'))
        pickle.dump(cum_reward, open('cumu_reward.p', 'wb'))

#save models
agent.q_eval.save_checkpoint()
agent.q_next.save_checkpoint()

plt.plot(per_episode_score)
plt.imshow(plt.show())
plt.savefig('per_epi_score.png')
    
plt.plot(cum_reward)
plt.show()
plt.savefig('Cumulative_Reward.png')
