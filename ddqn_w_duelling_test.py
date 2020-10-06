import gym
import numpy as np 
from ddqn_w_duelling import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(lr=0.005, gamma=0.99, n_actions=4, epsilon=1.0,
                    batch_size=64, input_dims=[8])
    n_games = 500
    scores = []
    eps_history = []

    for i in range(n_games):
        done=False
        score=0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            observation = next_observation
            agent.learn()
            # env.render()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('Episode: ',i,' Score: %.1f' %score, 'Avg. score: %.1f' %avg_score)

    plt.plot(scores)
    plt.plot(eps_history)
    plt.show()
