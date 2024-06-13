# Metasurface design with DQN

This is an extension of our [previous study](https://github.com/jLabKAIST/Physics-Informed-Reinforcement-Learning).

The RL environment used for this experiment follows `gym<0.25` API, therefore, no truncation.


## Setup
This example requires seperate Python environment, since this study was carried out some time ago.

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Training
### Run
You can run training process with below command.
Refer to `main.py` for RLlib related arguments. 

Especially, if `num_cpus_per_worker*num_rollout_workers` exceeds your hardware spec, an error will be raised.
```shell
python main.py --num_cpus_per_worker {desired-cpu-cores} --num_rollout_workers {desired-num-workers}
```