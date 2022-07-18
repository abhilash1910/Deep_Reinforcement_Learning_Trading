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




