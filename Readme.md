## Session on Deep Reinforcement Learning based Trading

DRL is currently being investigated in the area of algorithmic trading, and forecasting because of the
adaptability of algorithms in diverse environments. DRL follows 2 major approaches:

- Model Free RL
- Model Based RL

<img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg">


In generic DRL, the trading Agent is responsible for executing all call actions (buy, hold, sell) and is sometimes moderated by other Agents. There can be several variations in Agent based DRL which includes mutual collaboration across Agents, dueling or competition between the Agents, to creating diverse complex Environments. In this segment, we will be focussing on creating Agents which can identify to take actions - buy,hold and sell depending on the current share prices, treasury rates and penalties. The Agent gets a reward if it is able to attain a profit at the end of a trading period and is rewarded with a penalty in all cases (including cases where opportunities were missed). Model Free RL encompasses the class of algorithms which focus on a specific set of goals and performs optimization of the reward function, Agent brains and Environment parameters. Model Free RL again comprises of 3 centralized class of algorithms:

 - Off Policy Algorithms: Deep Q Network based algorithms (eg: DQN)
 - On Policy Algorithms: Actor Critic based algorithms (eg: A2C)
 - On-Off Policy Algorithms: Combined Actor Critics with experience replay algorithms (eg: DDPG)

We will be focussing on these variations of Model Free RL for creating different Agents and variations for our use case. We will start off with Off policy algorithms

### Off Policy

 Methods in this family learn an approximator Q_{\theta}(s,a) for the optimal action-value function, Q**(s,a). Typically they use an objective function based on the [Bellman equation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations). This optimization is almost always performed off-policy, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained. The corresponding policy is obtained via the connection between Q^* and \pi^*: the actions taken by the Q-learning agent are given by:

 <img src="https://spinningup.openai.com/en/latest/_images/math/d353412962e458573b92aac78df3fbe0a10d998d.svg">

 Some points on Off Policy:

 - Off policy methods rely on experience replay or buffer memory for making future estimations of the Value function
 - Off policy estimates across state space of all possible values V.
 - Off policy requires single Agents without requiring explicit any policy for estimating Value function

 Off policy or discrete /continuous spaces require non linear optimization which tabular RL cannot solve. This is because the number of value states greatly increases in continuous action spaces which is not possible to enumerate through tabular off policy methods mentione above. Hence Deep Off policy methods like Deep Q Network comes into play. 
 
 
 A [DQN](https://paperswithcode.com/paper/playing-atari-with-deep-reinforcement), or Deep Q-Network, approximates a state-value function in a Q-Learning framework with a neural network. In the Atari Games case, they take in several frames of the game as an input and output state values for each action as an output.
 It is usually used in conjunction with Experience Replay, for storing the episode steps in memory for off-policy learning, where samples are drawn from the replay memory at random. Additionally, the Q-Network is usually optimized towards a frozen target network that is periodically updated with the latest weights every  steps (where  is a hyperparameter). The latter makes training more stable by preventing short-term oscillations from a moving target. The former tackles autocorrelation that would occur from on-line learning, and having a replay memory makes the problem more like a supervised learning problem.  In case of trading, the Agent has to decide which states return the best values with the environment being all the stock prices,treasury amounts,time limits and so on . With the correct environment settings, the DQN agent tries to maximise the proit gain by choosing the correct time to select a stock for selling conditioned on different trading parameters. Also in case the Agent fails to register a threshold profit in a particular trading cycle, penalty amounts are added to allow the Agent to forecast and use its "experience bufer" to correctly determine the appropriate time to buy,hold or sell a stock. The core DQN agent is present in [Agent.py](https://github.com/abhilash1910/Deep_Reinforcement_Learning_Trading/blob/master/Agent.py).

 
 <img src="https://miro.medium.com/max/1400/0*YJ2RwEPbfYag0srW">
 
 
 A [Double DQN](https://arxiv.org/abs/1509.06461v3) utilises Double Q-learning to reduce overestimation by decomposing the max operation in the target into action selection and action evaluation. We evaluate the greedy policy according to the online network, but we use the target network to estimate its value. The update is the same as for DQN, but replacing the target with:
 
![image](https://user-images.githubusercontent.com/30946547/194940316-a98276dd-6ae3-4172-858e-bcbd9b6f43f4.png)

Compared to the original formulation of Double Q-Learning, in Double DQN the weights of the second network are replaced with the weights of the target network for the evaluation of the current greedy policy. In the context of trading, the single DQN agent has 2 brains which are trying to evaluate a greedy policy through an online network. As in case of standard DQN, the replay buffer stores the recent store of events,rewards and actions which is then used by the brains to communicate on the next step. The DQN Agent is in [DDQN_Agent.py](https://github.com/abhilash1910/Deep_Reinforcement_Learning_Trading/blob/master/DDQN_Agent.py)




