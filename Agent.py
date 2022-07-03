# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:17:26 2022

@author: Abhilash
"""

import random
from collections import deque
import tensorflow
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

# https://arxiv.org/pdf/1312.5602.pdf
class DQN_Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        self.model = self.model()
        self.tensorboard = TensorBoard(log_dir='./logs/DQN')
        self.tensorboard.set_model(self.model)

    def model(self):
        model = Sequential()
        #model.add(Dense(units=128,input_dim=self.state_dim,activation='relu'))
        model.add(Dense(units=64,input_dim=self.state_dim,activation='relu'))
        model.add(Dense(units=32,activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim,activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        print(model.summary())
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

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
        env_step = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]
        for state, actions, reward, next_state, done in env_step:
            if not done:
                print("Target Q value not attained")
                Q_value = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                print("Target Q value attained")
                Q_value = reward
            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_value
            history = self.model.fit(state, next_actions, epochs=1, verbose=2)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]