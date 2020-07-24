# My RL Implementations

This repository has my implementations of different RL algorithms.

### Learning Resources:

- In case you're interested in getting started with RL, then I feel 
[this blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#q-learning-off-policy-td-control) is a great starting point.

- Along with the above post you should start watching, [CS-234:Reinforcement Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u) from Stanford.


### Installing dependencies:
```bash
# If you're using python2
python2 -m pip install -r requirements.txt

# If you're using python3
python3 -m pip install -r requirements.txt
```

### Table of contents:
- Agents:
    - [Q-Table Learning](./agents/q_learning.py)
    - [Deep Q Learning](./agents/deepq.py)


- Environments:
    - [Grid World](./envs/gridworld)
    - [Frozen Lake](./envs/frozenLake)