import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
import os

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, checkpt_dir='./policy_grad_models'):
        super(PolicyNetwork,self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'.pth')
    

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')#///
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def save_checkpoint(self):
        print("SAving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print("Loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):    #agent has a policy, not is a policy
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4,
                    l1_size=128, l2_size=128, name='test'):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(lr, input_dims, l1_size, l2_size, n_actions, name)
    
    def choose_action(self, observation):
        probs = F.softmax(self.policy.forward(observation),dim=0)
        action_probs = T.distributions.Categorical(probs)
        action_chosen = action_probs.sample()
        log_probs = action_probs.log_prob(action_chosen)
        self.action_memory.append(log_probs)

        return action_chosen.item()
    
    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()   #zero the gradients
        G = np.zeros_like(self.reward_memory, dtype=np.float64)

        for i in range(len(self.reward_memory)):
            G_total = 0
            discount = 1 #gamma^0=1
            for j in range(i,len(self.reward_memory)):
                G_total += self.reward_memory[j] * discount
                discount *= self.gamma
            G[i] = G_total
        
        #normalization of rewards : similar to normalizing in ML
        mean = np.mean(G)
        std_dev = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std_dev

        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -g * log_prob            
        #grad. ascent but loss taken for pytorch to backprop, --minimise loss, max. return
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
    
    def save_model(self):
        self.policy.save_checkpoint()
        print('Model saved....')
    
    def load_model(self):
        self.policy.load_checkpoint()
        print('Model LOADED!!!')
        self.policy.eval()
