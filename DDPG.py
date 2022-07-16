# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:46:52 2022

@author: Abhilash
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import threading
from utils import Portfolio

# Tensorflow GPU configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()


class ActorNetwork:
    def __init__(self, sess, state_size, action_dim, buffer_size, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.model, self.states = self.create_actor_network(state_size, action_dim)
        self.model_target, self.target_state = self.create_actor_network(state_size, action_dim)
        self.model_target.set_weights(self.model.get_weights()) 
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, action_dim])
        print("chain rule: ∂a/∂θ * ∂Q(s,a)/∂a (action_gradients); minus sign for gradient descent; 1/buffer_size for mean value")
        self.sampled_policy_grad = tf.gradients(self.model.output/buffer_size, self.model.trainable_weights, -self.action_gradient)
        self.update_actor_policy = Adam(learning_rate=learning_rate).apply_gradients(zip(self.sampled_policy_grad, self.model.trainable_weights))

    def train(self, states_batch, action_grads_batch):
        print("Policy gradient",self.update_actor_policy)
        self.update_actor_policy
        
    
    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(24, activation='relu')(states)
        h1 = Dense(48, activation='relu')(h0)
        h2 = Dense(24, activation='relu')(h1)
        actions = Dense(self.action_dim, activation='softmax')(h2)
        model = Model(inputs=states, outputs=actions)
        self.actor_model=model
        return self.actor_model, states


class CriticNetwork:
    def __init__(self, sess, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.model, self.actions, self.states = self.create_critic_network(state_size, action_dim)
        self.model_target, self.target_action, self.target_state = self.create_critic_network(state_size, action_dim)
        self.action_grads = tf.gradients(self.model.output, self.actions)

    def gradients(self, states_batch, actions_batch):
        return self.sess.run(self.action_grads, feed_dict={self.states: states_batch, self.actions: actions_batch})[0]

    
    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(24, activation='relu')(h0)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        Q = Dense(action_dim, activation='relu')(h3)
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=1e-6))
        return model, actions, states



class DDPG_Agent(Portfolio):
    def __init__(self, state_dim, balance,is_eval=False):
        super().__init__(balance=balance)
        self.model_type = 'DDPG'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 90
        self.is_eval=False
        self.gamma = 0.95 # discount factor
        self.is_eval = is_eval
        tau = 0.001  # Target network hyperparameter
        self.tau=tau
        learning_rate_actor = 0.001  # learning rate for Actor network
        learning_rate_critic = 0.001  # learning rate for Critic network
        model_name="DDPG"
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995
        self.actor = ActorNetwork(sess, state_dim, self.action_dim, self.buffer_size, tau, learning_rate_actor, is_eval, model_name)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, tau, learning_rate_critic)
        self.actor_target = self.actor.model
        self.critic_target =self.critic.model
        self.actor_target.set_weights(self.actor.model.get_weights()) # hard copy model parameters to target model parameters
        self.critic_target.set_weights(self.critic.model.get_weights())
        
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/DDPG_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.critic.model)

    def reset(self):
        self.reset_portfolio()
        
    def remember(self, state, actions, reward, next_state, done):
    	self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state)[0]
        print("Action",actions)
        return actions
    
    def update_targets(self):
        critic_weights = self.critic.model.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)
        actor_weights = self.actor.model.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)
        
        

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        y_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = self.critic.model_target.predict([next_state, self.actor.model_target.predict(next_state)])
                y = reward + self.gamma * Q_target_value
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)

        y_batch = np.vstack(y_batch)
        states_batch = np.vstack([tup[0] for tup in mini_batch]) # batch_size * state_dim
        actions_batch = np.vstack([tup[1] for tup in mini_batch]) # batch_size * action_dim
        #lock=threading.Lock()
        #lock.acquire()
        # update critic by minimizing the loss
        loss = self.critic.model.train_on_batch([states_batch, actions_batch], y_batch)
        print("Critic Loss", loss)
        #lock.release()
        # update actor using the sampled policy gradients
        action_grads_batch = self.critic.gradients(states_batch, self.actor.model.predict(states_batch)) # batch_size * action_dim
        self.actor.train(states_batch, action_grads_batch)
        self.update_targets()
        if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
       
        # update target networks
        #self.actor.transfer_target()
        #print("Transfer weight to actor")
        #self.critic.train_target()
        
        return loss