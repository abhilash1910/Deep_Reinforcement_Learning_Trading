# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 00:57:58 2022

@author: Abhilash
"""

import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from utils import Portfolio



# https://arxiv.org/pdf/1802.09477.pdf
# https://arxiv.org/pdf/1509.06461.pdf
# https://papers.nips.cc/paper/3964-double-q-learning.pdf
class DDQN_Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False):
        super().__init__(balance=balance)
        self.model_type = 'DDQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval

        self.model = self.model()
        self.model_target =self.model
        self.model_target.set_weights(self.model.get_weights()) # hard copy model parameters to target model parameters

        self.tensorboard = TensorBoard(log_dir='./logs/DDQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = tf.keras.backend.abs(error) <= clip_delta
        squared_loss = 0.5 * tf.keras.backend.square(error)
        quadratic_loss = 0.5 * tf.keras.backend.square(clip_delta) + clip_delta * (tf.keras.backend.abs(error) - clip_delta)
        return tf.keras.backend.mean(tf.where(cond, squared_loss, quadratic_loss))
    
    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def model(self):
        #Q=V
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        #model.compile(loss=self._huber_loss, optimizer=Adam(lr=0.001))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        print("Append state to memory ")
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            print("random value is less than epsilon")
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        print("Select online from buffer")
        mini_batch = random.sample(self.memory, self.buffer_size)

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                print("Target Q value not attained")
                Q_value = reward + (1 - done) * self.gamma * np.amax(self.model_target.predict(next_state)[0])
            else:
                print("Target Q value attained")
                Q_value=reward
            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_value
            
            history = self.model.fit(state, next_actions, epochs=1, verbose=2)
            self.update_model_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]