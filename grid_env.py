import numpy as np


NUM_ROWS = 4
NUM_COL = 4
START_STATE = (0,0)
MAX_ACTIONS = 6
DETERMINISTIC = True
EXP_RATE = 0.5

REWARDS = { (1,1):1,
            (3,1):5,
            (3,3):10,
            (0,3):2,
            (0,0):0
          }

# print(REWARDS[(1,1)])

class GridWorldEnv():
    def __init__(self):
        self.state = START_STATE
        self.action = ""
        self.exp_rate = EXP_RATE
        self.grid = np.zeros([NUM_COL,NUM_ROWS])
        

    def printGrid(self):
        # self.grid[0][0] = '1'
        # self.state = np.array(list(self.state)) + np.array([1,1])
        self.grid[self.state[0]][self.state[1]] = 1
        print(self.grid)

    def nxtPosition(self):
        pass

    def giveReward(self):
        return REWARDS[self.state]

    def chooseAction(self):
        # exploration vs exploitation
        pass

if __name__ == "__main__":
    env = GridWorldEnv()
    env.printGrid()
    # print(env.giveReward())