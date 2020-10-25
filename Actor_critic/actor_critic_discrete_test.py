import gym
import numpy as np 
from actor_critic_discrete import Agent
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    agent = Agent(ALPHA=0.0001, input_dims=[8], GAMMA=0.99, n_actions=4, fc1_size=2048, fc2_size=248)

    env = gym.make('LunarLander-v2')
    score_history = []
    num_games = 2000

    for i in range(num_games):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, reward, next_state, done)
            state = next_state
            score+=reward
        
        score_history.append(score)
        print("Episode: ", i, "Score %.2f" %score)
    plt.plot(score_history)
    plt.savefig('AC_Discrete_LunarLander-v2.png')