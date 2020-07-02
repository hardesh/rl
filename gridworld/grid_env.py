import numpy as np
from copy import deepcopy

# Env Parameters
NUM_ROWS = 4
NUM_COL = 4
START_STATE = np.array((0, 0))
MAX_ACTIONS = 6
DETERMINISTIC = True
EXP_RATE = 0.5

# Dictionaries can only hash immutable objects
REWARDS = {(1, 1): 1, (3, 1): 5, (3, 3): 10, (0, 3): 2, (0, 0): 0}

ACTIONS = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]


class GridWorldEnv:
    """
    A grid world env for training an RL agent

    Attributes:
        actions (dict): maps the actions to change in state after taking that action
        exp_rate (float): Exploration rate of the agent
        state (np.array): Position [x,y] of the agent in the gridworld
        action (string): An action from [up,down,left,right]
        grid (np.array): The environment
        action_num (int): Number of actions taken by the agent
    """

    def __init__(self, exp_rate=EXP_RATE):
        self.actions = {"up": (0, -1), "down": (-1, 0), "left": (0, 1), "right": (1, 0)}
        self.exp_rate = exp_rate
        self.action_space = len(self.actions.keys())

        self.state = np.array(START_STATE)
        self.action = ""
        self.grid = np.zeros([NUM_COL, NUM_ROWS])
        self.action_num = 0

        self.observation_space = self.grid.size

    def reset(self):
        """
        Resets the environment to its original postion
        """
        self.state = np.array(START_STATE)
        self.action = ""
        self.grid = np.zeros([NUM_COL, NUM_ROWS])
        self.action_num = 0

    def render(self):
        """
        Prints the grid on the terminal
        """
        print("Present Grid: ")
        self.grid[self.state[0]][self.state[1]] = self.action_num
        print(self.grid.T)
        print()

    def isInvalid(self, state):
        """
        Returns true if the given state is valid
        """
        return state[0] < 0 or state[1] < 0 or state[0] > 3 or state[1] > 3

    def step(self, action, update=False):
        """
        Always gives valid next state no matter what the current state is 
        
        Args:
            action (str): A string which describes the action taken by the agent [up,down,left,right]
            update (bool): A boolean which decides whether to update the current state

        Returns:
            state (np.array): A state after after taking the given passed to this function
        """
        next_state = deepcopy(self.state)
        next_state = next_state + self.actions[action]
        if self.isInvalid(next_state):
            return self.state
        else:
            if update is True:
                self.state = next_state
                self.action_num += 1
            return next_state

    def getReward(self, state):
        """
        Gives rewards for a particular state

        Args:
            state (np.array): A valid state in the gridworld

        Returns:
            reward (int): An integer reward for being in a certain state
        """
        try:
            return REWARDS[tuple(state)]
        except KeyError:
            return 0

    def chooseAction(self):
        """
        Chooses the best action based on the exploration rate

        This function should be in the learning algorithm
        """
        max_next_reward = 0
        action = ""

        if np.random.uniform(0, 1) < self.exp_rate:
            # exploration
            action = np.random.choice(list(self.actions.keys()))
        else:
            # exploitation: choosing the action which gives most reward. Assuming agent can only see the next state
            for a in self.actions.keys():
                next_reward = self.getReward(self.step(a))

                if next_reward >= max_next_reward:
                    max_next_reward = next_reward
                    action = a

        return action


if __name__ == "__main__":
    env = GridWorldEnv()
    env.render()
    # print(env.getReward(env.state))
    for i in range(6):
        action = env.chooseAction()
        env.state = env.step(action, update=True)
        env.render()

    Q = np.zeros((env.observation_space, env.action_space))
    print(np.argmax(Q[[0,0],:]))