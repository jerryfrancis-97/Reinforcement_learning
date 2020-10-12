import gym
import numpy as np 
from policy_grad import Agent
import matplotlib.pyplot as plt
from gym import wrappers #for saving video of training

#for ffmpeg n UBuntu19.04 : https://www.linuxhelp.com/how-to-install-ffmpeg-on-ubuntu-19-04


if __name__ == '__main__':
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    '''
    agent = Agent(lr=0.001, input_dims=[8], gamma=0.99, n_actions=4,
                    l1_size=128, l2_size=128, name='LunarLander-v2-1000eps')
    
    score_history = []
    score = 0
    games = 1000

    env = wrappers.Monitor(env, 'LunarLander-train', video_callable=lambda episode_id: True, force=True)

    for i in range(games):
        print('Episode: ',i,'score: ',score)
        score = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            score += reward
            obs = next_obs
        score_history.append(score)
        agent.learn()
    
    agent.save_model()
    plt.plot(score_history, label="Scores")
    plt.savefig('policy_grad(LunarLander-v2)_1000eps.png')
'''
    print("---->    PLaying learned model....")
    good_agent = Agent(lr=0.001, input_dims=[8], gamma=0.99, n_actions=4,
                    l1_size=128, l2_size=128, name='LunarLander-v2-1000eps')
    good_agent.load_model()
    test_games = 10
    test_score = 0
    test_history = []

    env = wrappers.Monitor(env, 'LunarLander_test_results', video_callable=lambda episode_id: True, force=True)

    for i in range(test_games):
        test_score = 0
        obs = env.reset()
        done = False
        while not done:
            action = good_agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            good_agent.store_rewards(reward)
            test_score += reward
            obs = next_obs
            env.render()
        print('Episode: ',i,'score: ',test_score)
        test_history.append(test_score)
    
    plt.plot(test_history, label="Scores")
    plt.savefig('policy_grad(LunarLander-v2)_1000eps_test_results.png')
