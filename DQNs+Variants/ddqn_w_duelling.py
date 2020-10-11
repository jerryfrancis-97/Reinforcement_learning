import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np 

class Duelling_DDQN(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(Duelling_DDQN,self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')   #fcw_dims are output dims

        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q
    
    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

class Replaybuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype= np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype= np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype= np.int32)
        self.terminal_state_memory = np.zeros(self.mem_size, dtype= np.bool)
    
    def store_transition(self, state, action, reward, next_state, done):
        index  =  self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_state_memory[index] = done

        self.mem_counter +=1
    
    def sample(self, batch_size):
        max_size = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_size, batch_size, replace=True)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_state_memory[batch]

        return states, actions, rewards, next_states, dones


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3,
                    epsilon_min=0.01, mem_size=1000000, fc1_dims=64, fc2_dims=64, replace=1000):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.replace = replace
        self.batch_size = batch_size
        
        self.learn_step_counter = 0
        self.memory = Replaybuffer(mem_size, input_dims)
        self.q_eval = Duelling_DDQN(n_actions, fc1_dims, fc2_dims)
        self.q_next = Duelling_DDQN(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                loss='mean_squared_error')
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        
        return action
    
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        if self.learn_step_counter % self.replace ==0:
            self.q_next.set_weights(self.q_eval.get_weights())
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(next_states)
        # changing q_pred doesn't matter as we are passing states to train func. anyway.

        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(next_states), axis=1)

        #simple soln.
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx, max_actions[idx]] * (1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target) #states->q_eval->actual_got_q-values wrt. q_target(calc.)

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter+=1

        
        