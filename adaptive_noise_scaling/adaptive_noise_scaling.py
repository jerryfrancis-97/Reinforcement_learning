import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

class My_policy():
    def __init__(self, state_size, action_size,noise=1e-4):
        self.noise_scale = noise
        self.weights = 1e-4 * np.random.rand(state_size,action_size)

    def forward(self,state):
        x = np.dot(state, self.weights)
        return x

    def make_noisy(self,x,inc_range):
        if inc_range:  #inc. range
            self.noise_scale = min(2,self.noise_scale * 2)
            return x + self.noise_scale * np.random.rand(*x.shape)
        else:
            self.noise_scale = max(1e-3,self.noise_scale / 2)
            x += self.noise_scale * np.random.rand(*x.shape)
            return x

    def softmax(self, x):
        return np.exp(x)/sum(np.exp(x))
    
    def choose_action(self, observation):
        action_vals = self.forward(observation)
        action_probs = self.softmax(action_vals)
        chosen_action = np.argmax(action_probs)
        return chosen_action

my_parser = argparse.ArgumentParser(description='List env., state and action space...')
my_parser.add_argument('--Env',required=True, type=str, help='Mention gym environment name')
my_parser.add_argument('--state',required=True, type=int, help='Mention state size')
my_parser.add_argument('--action',required=True, type=int, help='Mention action size')

args = my_parser.parse_args()
env = gym.make(args.Env)
env.seed(0)
np.random.seed(0)

state_size = args.state
action_size = args.action
Agent = My_policy(state_size, action_size, 1e-2)

games = 10000
gamma = 0.99
returns = []
scores = []
noise_rad_history = []
best_return = -np.Inf
best_weight = Agent.weights

for i in range(games):
    done = False
    rewards = []
    observation = env.reset()
    while not done:
        action = Agent.choose_action(observation)
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)
        observation = next_obs
        # env.render()
    scores.append(sum(rewards))

    discount = [gamma**i for i in range(len(rewards)+1)]
    # print('discount: ',len(discount),'rewards: ',len(rewards))
    R = sum([a*b for a,b in zip(discount,rewards)])
    returns.append(R)
    noise_rad_history.append(Agent.noise_scale)

    if R>=best_return:
        best_weight = Agent.weights
        best_return = R 
        Agent.weights = Agent.make_noisy(Agent.weights, False)    #decrease radiius
    else:
        Agent.weights = Agent.make_noisy(best_weight, True)

    avg_score = np.mean(scores[-100:])  
    print('Episode: ',i,'Score: %.1f'%scores[-1], 'Avg_score: %.2f'%avg_score, 'Return: %.1f'%R)
    if avg_score >= 195.0:
        print("Env solved in ",i-100,'episodes.')
        print("Best return : ",best_return)
        print()
        print("Best policy: ",best_weight)
        break



plt.plot(np.arange(1,len(scores)+1),scores, label='Scores')
plt.plot(np.arange(1,len(scores)+1),returns, label='Expctd. returns')
plt.plot(np.arange(1,len(scores)+1),[n*10 for n in noise_rad_history], label='Noise(10x)')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()
plt.savefig('adapt_noise('+args.Env+').png')
