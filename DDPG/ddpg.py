import os
import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 


class OUAction_Noise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        # working : noise=OUAction_Noise() ; noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0 
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)
    
    def store_transition(self, state, action, reward, next_state, done):
        index  = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = 1-done
        self.mem_counter+=1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='critic_temp/ddpg'):
        super(CriticNetwork,self).__init()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #initialisation
        temp1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -temp1, temp1)
        T.nn.init.uniform_(self.fc1.bias.data, -temp1, temp1)
        #batch_norm
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #initialisation
        temp2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -temp2, temp2)
        T.nn.init.uniform_(self.fc2.bias.data, -temp2, temp2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        temp3 = 0.002
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -temp3, temp3)
        T.nn.init.uniform_(self.q.bias.data, -temp3, temp3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = F.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)

        action_val = F.relu(self.action_value(action))
        state_action_val = F.relu(T.add(state_val, action_val))
        q_val = self.q(state_action_val)

        return q_val
    
    def save_checkpoint(self):
        print(".....saving checkpoint.....")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(".... loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='actor_temp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        t1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -t1,t1)
        T.nn.init.uniform_(self.fc1.bias.data, -t1,t1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.002
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self,  state):
        state = self.fc1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = T.tanh(self.mu(state))

        return state
    
    def save_checkpoint(self):
        print(".....saving checkpoint.....")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(".... loading checkpoint....")
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99
                    n_actions=2, max_size=1000000, layer1_size=400,
                    layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, 
                                            n_actions=n_actions, name='TargetActor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                            layer2_size, n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                            layer2_size, n_actions, name='TargetCritic')
        self.noise = OUAction_Noise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        self.actor.train()
        return mu_prime.cpu().detach.numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        next_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state,action)

        target = []
        for i in range(self.batch_size):
            target.append( reward[i] + self.gamma * next_critic_value[i] * done[i] )
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size,1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critc.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self. tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        #update target critic
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() +
                                        (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        #update actor critic
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() +
                                        (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_crtiic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_crtiic.load_checkpoint()

                

       

    

        

