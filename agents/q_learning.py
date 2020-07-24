import numpy as np


class QLearning:
    """
    Q - Learning Algorithm

    Attributes:
        env (gym.Env): environment
        epsilon (float): parameter for epsilon-greedy exploration
        gamma (float): discount factor
        lr (float): learning rate
    """

    def __init__(self, env, epsilon, lr, gamma):
        self.env = env
        self.lr = lr
        self.epsilon = epsilon  # for epsilon greedy policy selection
        self.gamma = gamma

        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def get_action(self, state, explore=True):
        if explore:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()

        return np.argmax(self.Q[state, :])

    def update(self, sa_tuple):
        """
        Updates the Q-table

        Args:
            sa_tuple (Tuple): A state-action tuple with the following elements
                            (state, action, reward, next_state)
        """
        state, action, reward, next_state = sa_tuple
        self.Q[state, action] += self.lr * (
            reward + np.max(self.Q[next_state, :]) - self.Q[state, action]
        )
