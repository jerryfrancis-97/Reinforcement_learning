import gym
import numpy as np 
from duelling_dqn import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
env = gym.make('LunarLander-v2')
games = 1000
load_checkpoint = False

agent = Agent(gamma=0.99, epsilon=1.0, alpha=4e-3, input_dims=[8],
                n_actions=4, mem_size=10000, eps_min=0.01,
                batch_size=64, eps_rate=1e-3, change2target=100)

if load_checkpoint:
    agent.load_models()

scores = []
eps_history = []

for i in range(games):  
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, next_observation, done)
        agent.learn()
        observation = next_observation
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print("Episode_",i),": score: %.1f, average_score: %.1f, epsilon: %.2f" %(score,avg_score,agent.epsilon))
    eps_history.append(agent.epsilon)
    
