import numpy as np

class Agent:

    def __init__(self, state_space=None, act_space=None, epsilon=0.1, learning_rate=0.001, gamma=0.99):
        if state_space is None or act_space is None:
            print("agent constructor: missing arguments")
            exit(0)
        self.state_space = state_space
        self.act_space = act_space
        shape = []
        for var in self.state_space:
            shape.append(var)
        shape.append(self.act_space)
        self.Q = np.zeros(shape)

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

    def choose_action(self, state):
        p = np.random.uniform()
        if p < self.epsilon:
            return np.random.randint(0, self.act_space)
        else:
            return np.argmax(self.Q[state])

    def train(self, s, a, r, s1):

        td_error = r + self.gamma * np.max(self.Q[s1]) - self.Q[s + (a,)]
        self.Q[s + (a,)] += self.learning_rate * td_error

    def schedule_exploration(self, decay=0.99999):
        self.epsilon = max(0.01, self.epsilon*decay)
