# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 13:12:25 2022

@author: Abhilash
"""

import argparse
import importlib
import logging
import sys
import time
import numpy as np
from utils import *
from Agent import *
from DDQN_Agent import *
from DuelingDDQN_Agent import *
from AC_Agent import *
from Hard_A2C import *
from TRPO_A2C import *
from DDPG import *

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--stock_name', action="store", dest="stock_name", default='S&P_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

#model_name="DQN"
#model_name="DDQN"
#model_name="DuelingDDQN"
#model_name="AC"
#model_name="Hard_A2C"
#model_name="TRPO_A2C"
model_name="DDPG"
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
delta=1e-7
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

#agent = DQN_Agent(state_dim=window_size + 3, balance=initial_balance)
#agent=DDQN_Agent(state_dim=window_size + 3, balance=initial_balance)
#agent=DuelingDDQN_Agent(state_dim=window_size + 3, balance=initial_balance)
#agent=AC_Agent(state_dim=window_size + 3, balance=initial_balance)
#agent=Hard_A2C_Agent(state_dim=window_size + 3, balance=initial_balance)
#agent=TRPO_A2C_Agent(state_dim=window_size + 3, balance=initial_balance)
agent=DDPG_Agent(state_dim=window_size + 3, balance=initial_balance)

def hold(actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)

# configure logging
logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()
for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset() # reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        
        if model_name == 'AC' or model_name=='Hard_A2C' or model_name=="TRPO_A2C" or model_name=='DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
        
            actions = agent.model.predict(state)[0]
            action = agent.act(state)
        
        # execute position
        logging.info('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0: # hold
            execution_result = hold(actions)
        if action == 1: # buy
            execution_result = buy(t)      
        if action == 2: # sell
            execution_result = sell(t)        
        
        # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple): # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
            logging.info(execution_result)                

        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit+delta

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            print("Getting Loss")
            loss = agent.experience_replay()
            logging.info('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
        elif model_name=='DDQN':
            agent.model.save('saved_models/DDQN_ep' + str(e) + '.h5')
        elif model_name=='DuelingDDQN':
            agent.model.save('saved_models/DuelingDDQN_ep' + str(e) + '.h5')
        
        #tbd-> on policy
        elif model_name == 'AC':
            agent.actor.model.save_weights('saved_models/AC_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/AC_ep{}_critic.h5'.format(str(e)))
        elif model_name == 'Hard_A2C':
            agent.actor.model.save_weights('saved_models/A2C_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/A2C_ep{}_critic.h5'.format(str(e)))
        elif model_name=="TRPO_A2C":
            agent.actor.model.save_weights('saved_models/TRPO_A2C_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/TRPO_A2C_ep{}_critic.h5'.format(str(e)))
        elif model_name=="DDPG":
            agent.actor.model.save_weights('saved_models/DDPG_A2C_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/DDPG_A2C_ep{}_critic.h5'.format(str(e)))
        
        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)