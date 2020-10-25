import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 

class Single_Actor_Critic(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Single_Actor_Critic,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, slf.fc2_dims)
        self.actor = nn.Linear(self.fc2_dims, self.n_actions)
        self.critic = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        state = T.tensor(x).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)
        v = self.critic(x)

        return policy, v
    
class Agent():
    def __init__(self, ALPHA, input_dims, GAMMA=0.99, fc1_size=128, fc2_size=128, n_actions=2):
        self.GAMMA = GAMMA
        self.actor_critic = Single_Actor_Critic(ALPHA, input_dims, fc1_size,fc2_size,n_actions=n_actions)
        self.log_probab = None
    
    def choose_action(self, state):
        action_prob, _ = self.actor_critic.forward(state)
        curr_policy = F.softmax(action_prob, dim=0)
        action_dist = T.distributions.Categorical(curr_policy)
        action = action_dist.sample()
        self.log_probab = action_dist.log_prob(action)

        return action.item()
    
    def learn(self, state, reward, next_state, done):
        self.actor_critic.optimizer.zero_grad()
        _, state_value = self.actor_critic.forward(state)
        _, next_state_value = self.actor_critic.forward(next_state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        TD = reward + self.GAMMA * next_state_value * (1-int(done)) - state_value

        actor_loss = -self.log_probab * TD
        critic_loss = TD**2 #as TD is error

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
        