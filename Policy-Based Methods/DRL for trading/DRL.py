#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#             <img src='./text_images/nvidia.png' width="200" height="450">
#         </td>
#         <td> & </td>
#         <td>
#             <img src='./text_images/udacity.png' width="350" height="450">
#         </td>
#     </tr>
# </table>

# # Deep Reinforcement Learning for Optimal Execution of Portfolio Transactions     

# # Introduction
# 
# This notebook demonstrates how to use Deep Reinforcement Learning (DRL) for optimizing the execution of large portfolio transactions. We begin with a brief review of reinforcement learning and actor-critic methods.  Then, you will use an actor-critic method to generate optimal trading strategies that maximize profit when liquidating a block of shares. 
# 
# # Actor-Critic Methods
# 
# In reinforcement learning, an agent makes observations and takes actions within an environment, and in return it receives rewards. Its objective is to learn to act in a way that will maximize its expected long-term rewards. 
# 
# <br>
# <figure>
#   <img src = "./text_images/RL.png" width = 80% style = "border: thin silver solid; padding: 10px">
#       <figcaption style = "text-align: center; font-style: italic">Fig 1. - Reinforcement Learning.</figcaption>
# </figure> 
# <br>
# 
# There are several types of RL algorithms, and they can be divided into three groups:
# 
# - **Critic-Only**: Critic-Only methods, also known as Value-Based methods, first find the optimal value function and then derive an optimal policy from it. 
# 
# 
# - **Actor-Only**: Actor-Only methods, also known as Policy-Based methods, search directly for the optimal policy in policy space. This is typically done by using a parameterized family of policies over which optimization procedures can be used directly. 
# 
# 
# - **Actor-Critic**: Actor-Critic methods combine the advantages of actor-only and critic-only methods. In this method, the critic learns the value function and uses it to determine how the actor's policy parramerters should be changed. In this case, the actor brings the advantage of computing continuous actions without the need for optimization procedures on a value function, while the critic supplies the actor with knowledge of the performance. Actor-critic methods usually have good convergence properties, in contrast to critic-only methods.  The **Deep Deterministic Policy Gradients (DDPG)** algorithm is one example of an actor-critic method.
# 
# <br>
# <figure>
#   <img src = "./text_images/Actor-Critic.png" width = 80% style = "border: thin silver solid; padding: 10px">
#       <figcaption style = "text-align: center; font-style: italic">Fig 2. - Actor-Critic Reinforcement Learning.</figcaption>
# </figure> 
# <br>
# 
# In this notebook, we will use DDPG to determine the optimal execution of portfolio transactions. In other words, we will use the DDPG algorithm to solve the optimal liquidation problem. But before we can apply the DDPG algorithm we first need to formulate the optimal liquidation problem so that in can be solved using reinforcement learning. In the next section we will see how to do this. 

# # Modeling Optimal Execution as a Reinforcement Learning Problem
# 
# As we learned in the previous lessons, the optimal liquidation problem is a minimization problem, *i.e.* we need to find the trading list that minimizes the implementation shortfall. In order to solve this problem through reinforcement learning, we need to restate the optimal liquidation problem in terms of **States**, **Actions**, and **Rewards**. Let's start by defining our States.
# 
# ### States
# 
# The optimal liquidation problem entails that we sell all our shares within a given time frame. Therefore, our state vector must contain some information about the time remaining, or what is equivalent, the number trades remaning. We will use the latter and use the following features to define the state vector at time $t_k$:
# 
# 
# $$
# [r_{k-5},\, r_{k-4},\, r_{k-3},\, r_{k-2},\, r_{k-1},\, r_{k},\, m_{k},\, i_{k}]
# $$
# 
# where:
# 
# - $r_{k} = \log\left(\frac{\tilde{S}_k}{\tilde{S}_{k-1}}\right)$ is the log-return at time $t_k$
# 
# 
# - $m_{k} = \frac{N_k}{N}$ is the number of trades remaining at time $t_k$ normalized by the total number of trades.
# 
# 
# - $i_{k} = \frac{x_k}{X}$ is the remaining number of shares at time $t_k$ normalized by the total number of shares.
# 
# The log-returns capture information about stock prices before time $t_k$, which can be used to detect possible price trends. The number of trades and shares remaining allow the agent to learn to sell all the shares within a given time frame. It is important to note that in real world trading scenarios, this state vector can hold many more variables. 
# 
# ### Actions
# 
# Since the optimal liquidation problem only requires us to sell stocks, it is reasonable to define the action $a_k$ to be the number of shares to sell at time $t_{k}$. However, if we start with millions of stocks, intepreting the action directly as the number of shares to sell at each time step can lead to convergence problems, because, the agent will need to produce actions with very high values. Instead, we will interpret the action $a_k$ as a **percentage**. In this case, the actions produced by the agent will only need to be between 0 and 1. Using this interpretation, we can determine the number of shares to sell at each time step using:
# 
# $$
# n_k = a_k \times x_k
# $$
# 
# where $x_k$ is the number of shares remaining at time $t_k$.
# 
# ### Rewards
# 
# Defining the rewards is trickier than defining states and actions, since the original problem is a minimization problem. One option is to use the difference between two consecutive utility functions. Remeber the utility function is given by:
# 
# $$
# U(x) = E(x) + λ V(x)
# $$
# 
# After each time step, we compute the utility using the equations for $E(x)$ and $V(x)$ from the Almgren and Chriss model for the remaining time and inventory while holding parameter λ constant. Denoting the optimal trading trajectory computed at time $t$ as $x^*_t$, we define the reward as: 
# 
# $$
# R_{t} = {{U_t(x^*_t) - U_{t+1}(x^*_{t+1})}\over{U_t(x^*_t)}}
# $$
# 
# Where we have normalized the difference to train the actor-critic model easier.

# # Simulation Environment
# 
# In order to train our DDPG algorithm we will use a very simple simulated trading environment. This environment simulates stock prices that follow a discrete arithmetic random walk and that the permanent and temporary market impact functions are linear functions of the rate of trading, just like in the Almgren and Chriss model. This simple trading environment serves as a starting point to create more complex trading environments. You are encouraged to extend this simple trading environment by adding more complexity to simulte real world trading dynamics, such as book orders, network latencies, trading fees, etc... 
# 
# The simulated enviroment is contained in the **syntheticChrissAlmgren.py** module. You are encouraged to take a look it and modify its parameters as you wish. Let's take a look at the default parameters of our simulation environment. We have set the intial stock price to be $S_0 = 50$, and the total number of shares to sell to one million. This gives an initial portfolio value of $\$50$ Million dollars. We have also set the trader's risk aversion to $\lambda = 10^{-6}$.
# 
# The stock price will have 12\% annual volatility, a [bid-ask spread](https://www.investopedia.com/terms/b/bid-askspread.asp) of 1/8 and an average daily trading volume of 5 million shares. Assuming there are 250 trading days in a year, this gives a daily volatility in stock price of $0.12 / \sqrt{250} \approx 0.8\%$. We will use a liquiditation time of $T = 60$ days and we will set the number of trades $N = 60$. This means that $\tau=\frac{T}{N} = 1$ which means we will be making one trade per day. 
# 
# For the temporary cost function we will set the fixed cost of selling to be 1/2 of the bid-ask spread, $\epsilon = 1/16$. we will set $\eta$ such that for each one percent of the daily volume we trade, we incur a price impact equal to the bid-ask
# spread. For example, trading at a rate of $5\%$ of the daily trading volume incurs a one-time cost on each trade of 5/8. Under this assumption we have $\eta =(1/8)/(0.01 \times 5 \times 10^6) = 2.5 \times 10^{-6}$.
# 
# For the permanent costs, a common rule of thumb is that price effects become significant when we sell $10\%$ of the daily volume. If we suppose that significant means that the price depression is one bid-ask spread, and that the effect is linear for smaller and larger trading rates, then we have $\gamma = (1/8)/(0.1 \times 5 \times 10^6) = 2.5 \times 10^{-7}$. 
# 
# The tables below summarize the default parameters of the simulation environment

# In[1]:


import utils

# Get the default financial and AC Model parameters
financial_params, ac_params = utils.get_env_param()


# In[2]:


financial_params


# In[3]:


ac_params


# # Reinforcement Learning
# 
# In the code below we use DDPG to find a policy that can generate optimal trading trajectories that minimize implementation shortfall, and can be benchmarked against the Almgren and Chriss model. We will implement a typical reinforcement learning workflow to train the actor and critic using the simulation environment. We feed the states observed from our simulator to an agent. The Agent first predicts an action using the actor model and performs the action in the environment. Then, environment returns the reward and new state. This process continues for the given number of episodes. To get accurate results, you should run the code at least 10,000 episodes.

# In[4]:


import numpy as np

import syntheticChrissAlmgren as sca
from ddpg_agent import Agent

from collections import deque

# Create simulation environment
env = sca.MarketEnvironment()

# Initialize Feed-forward DNNs for Actor and Critic models. 
agent = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(), random_seed=0)

# Set the liquidation time
lqt = 60

# Set the number of trades
n_trades = 60

# Set trader's risk aversion
tr = 1e-6

# Set the number of episodes to run the simulation
episodes = 10000

shortfall_hist = np.array([])
shortfall_deque = deque(maxlen=100)

for episode in range(episodes): 
    # Reset the enviroment
    cur_state = env.reset(seed = episode, liquid_time = lqt, num_trades = n_trades, lamb = tr)

    # set the environment to make transactions
    env.start_transactions()

    for i in range(n_trades + 1):
      
        # Predict the best action for the current state. 
        action = agent.act(cur_state, add_noise = True)
        
        # Action is performed and new state, reward, info are received. 
        new_state, reward, done, info = env.step(action)
        
        # current state, action, reward, new state are stored in the experience replay
        agent.step(cur_state, action, reward, new_state, done)
        
        # roll over new state
        cur_state = new_state

        if info.done:
            shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
            shortfall_deque.append(info.implementation_shortfall)
            break
        
    if (episode + 1) % 100 == 0: # print average shortfall over last 100 episodes
        print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))        

print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(np.mean(shortfall_hist)))


# # Todo
# 
# The above code should provide you with a starting framework for incorporating more complex dynamics into our model. Here are a few things you can try out:
# 
# - Incorporate your own reward function in the simulation environmet to see if you can achieve a expected shortfall that is better (lower) than that produced by the Almgren and Chriss model.
# 
# 
# - Experiment rewarding the agent at every step and only giving a reward at the end.
# 
# 
# - Use more realistic price dynamics, such as geometric brownian motion (GBM). The equations used to model GBM can be found in section 3b of this [paper](https://ro.uow.edu.au/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=aabfj)
# 
# 
# - Try different functions for the action. You can change the values of the actions produced by the agent by using different functions. You can choose your function depending on the interpretation you give to the action. For example, you could set the action to be a function of the trading rate.
# 
# 
# - Add more complex dynamics to the environment. Try incorporate trading fees, for example. This can be done by adding and extra term to the fixed cost of selling, $\epsilon$.

# In[ ]:




