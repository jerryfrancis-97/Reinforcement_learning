import os
import torch as T
import torch.nn as nn
import torch.nn.F as F
import torch.optim as optim
import numpy as np 

class ReplayMemory(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.memory_space = np.zeros((self.mem_size, *input_shape), dtype= np.float32)  #*input_shape is list iterable
        self.new_memory_space = np.zeros((self.mem_size, *input_shape), dtype= np.float32)
        self.action_memory_space = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory_space = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory_space = np.zeros(self.mem_size,dtype=np.uint8)   #to store terminal state flags

    def storing_trans(self, state, action, reward, next_state, done):
        counter = self.mem_counter % self.mem_size #index of placing state

        self.memory_space[counter] = state
        self.new_memory_space[counter] = next_state
        self.action_memory_space[counter] = action
        self.reward_memory_space[counter] = reward
        self.terminal_memory_space[counter] = done

        self.mem_counter +=1
    
    def sample(self, batch_size):
        avail_size = min(batch_size, self.mem_size)
        batch = np.random.choice(avail_size, batch_size, replace=False) #samples w/o replacement

        states = self.memory_space[batch]
        actions = self.action_memory_space[batch]
        rewards = self.reward_memory_space[batch]
        next_states = self.new_memory_space[batch]
        dones = self.terminal_memory_space[batch]

        return states, actions, rewards, next_states, dones

class Duelling_DQN(nn.Module):
    def __init__(self, alpha, n_actions, name, input_dims, checkpt_dir='tmp/duel_dqn'):
        super(Duelling_DQN,self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128,128)

        self.V = nn.Linear(128,1)
        self.A = nn.Linear(28, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSEloss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name='duelling_dqn')
    
    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A
    
    def save_checkpoint(self):
        print("SAving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        print("Loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))
    
class Agent(object):
    def __init__(self, gamma, epsilon, alpha, n_actions, input_dims, mem_size,\
                    batch_size,eps_min=0.01, eps_rate=4e-8, change2target=1000,
                    checkpt_dir='tmp/duel_dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_rate = eps_rate
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0 #learning steps
        self.change2target = change2target
        
        self.memory = ReplayMemory(mem_size, input_shape, n_actions)
        self.q_val = Duelling_DQN(alpha, n_actions, input_dims=input_dims, name='q_val', checkpt_dir=checkpt_dir)
        self.q_next = Duelling_DQN(alpha, n_actions, input_dims=input_dims, name='q_next', checkpt_dir=checkpt_dir)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.storing_trans(state, action, reward, next_state, done)
    
    def choose_action(self, observation):
        if np.random.random()  > self.epsilon:
            observation = observation[np.newaxis,:]
            state = T.tensor(observation).to(self.q_val.device) #normal2tensor2gpu_variable
            _, advantage = self.q_val.forward(state)
            action = T.argmax(advantage).item() #tensor to number for gym
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def replace_target_network(self):
        if self.learn_step_counter % self.change2target == 0 and /
               self.change2target is not None:
            self.q_next.load_state_dict(self.q_val.state_dict())
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon - eps_rate if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self):
        #bsatch size fill
        if self.memory.mem_counter < self.batch_size:
            return
        
        self.q_val.optimiser.zero_grad()
        self.replace_target_network()

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = T.tensor(state).to(self.q_val.device)
        action = T.tensor(action).to(self.q_val.device)
        reward = T.tensor(reward).to(self.q_val.device)
        next_state = T.tensor(next_state).to(self.q_val.device)
        dones = T.tensor(done).to(self.q_val.device)
        # T.tensor(A) preserves dtype of variable A, T.Tensor() takes pytorch default dtype

        V_s, A_s = self.q_val.forward(state)
        V_next, A_next = self.q_next.forward(state)

        # Q = V + (A - avg(A))
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
        # for mean take action dim, keep dim same as before, gather the values along batch, action-unsqueeze==linear array
        
        q_next  = T.add(V_next, (A_next - A_next.mean(dim=1, keepdim=True)))
        q_target = reward + self.gamma*T.max(q_next, dim=1)[0].detach() #dim=1 as index is also returned, detach() to avoid backprop
        q_target[dones] = 0.0   #done=1 , episode is over, so no reward for q_target viz. 0

        loss = self.q_val.loss(q_pred, q_target).to(self.q_val.device)
        loss.backward() #backprop
        self.q_val.optimiser.step()
        
        self.learn_step_counter+=1
        self.decrease_epsilon()
        
    def save_models(self):
        self.q_val.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_val.load_checkpoint()
        self.q_next.load_checkpoint()

        
        





