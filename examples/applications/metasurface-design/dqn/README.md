# Metasurface design with DQN

This is an extension of our [previous study](https://github.com/jLabKAIST/Physics-Informed-Reinforcement-Learning).

The RL environment used for this experiment follows `gym<0.25` API, therefore, no truncation.


### Setup
This example requires seperate Python environment, since this study was carried out some time ago.

```shell
python3 -m venv dqn
source dqn/bin/activate
pip install -r requirements.txt
```
### DeflectorGym
`DeflectorGym` is our custom environment for RCWA simulation.
```shell
cd deflector-gym
pip install -e .
```

### Run
```shell
python main.py
```
