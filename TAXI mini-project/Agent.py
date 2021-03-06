import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, algorithm='sarsamax', start_epsilon=1, epsilon_decay=0.9, epsilon_cut=0.1, alpha=0.1, gamma=1,
                 nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon, self.epsilon_decay, self.epsilon_cut, self.alpha, self.gamma, self.nA = \
            start_epsilon, epsilon_decay, epsilon_cut, alpha, gamma, nA

    def select_action(self, state):
        r = random.random()
        if r > self.epsilon:   # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:  # otherwise, select an action randomly
            return random.randint(0, 5)

    def get_probs(self, Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s


    def step_exp_sarsa(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            probs = self.get_probs(self.Q[next_state], self.epsilon, self.nA)

            self.Q[state][action] += self.alpha * (
                        reward + self.gamma * np.dot(probs, self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon_cut is not None:
                self.epsilon = max(self.epsilon, self.epsilon_cut)


######   OR


    def step_sarsamax(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            self.Q[state][action] += self.alpha * (
                        reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon_cut is not None:
                self.epsilon = max(self.epsilon, self.epsilon_cut)
