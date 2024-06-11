# EM-neuraloperator
neural operator for approximating EM field

## Setup
```
pip install -r requirements.txt
```

Build neuraloperator from source.
We added BatchNorm in neuraloperator and logging is quite buggy (2024/03).
```
cd neuraloperator
pip install -e .
```

## Generate Dataset
```
python data.py
```

## Run
By default, hydra will pick `config.yaml` as configuration file.

Adjust the hyperparameters, etc in config file and run :
```
python main.py
```
With specified config file, run :
```
python main.py --config-name my_config.yaml
```
Sweep over hparams :
```
python main.py --config-name carsten.yaml --multirun seed=7,42,170 data=1100-70-4,1100-50-4,900-70-4,900-50-4
```

**Be sure** to fill experiment name at `experiement :` field.

This is practice to remind yourself what the purpose of this experiment is.
```
experiment: FNO-dense-l2-lowrank
device: cuda:1

model: NO3
model_config:
  n_modes: [32, 32]
  in_channels: 1
...
```
This will drop a folder with experiement name and datetime under `runs`.

In the folder, **.hydra** folder, **tensorboard** event file and **model** checkpoint will be saved.