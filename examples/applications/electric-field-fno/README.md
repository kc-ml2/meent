# Electric field prediction with Fourier neuarl opeartor

Code for training/testing baselines models for electric field prediction, with various losses.

We adapted the code from original author's [neuralop](https://github.com/neuraloperator/neuraloperator), and the process is controlled by [hydra](https://hydra.cc/docs/intro/).

## Setup
```
pip install -r requirements.txt
```

Build neuraloperator from source.
We added BatchNorm in neuraloperator, aggregation axis in loss function and the logging which is quite buggy (2024/03).
```
cd neuraloperator
pip install -e .
```

## Generate Dataset
For this experiment, we stick to simple pickle format, but for larger, complex datasets, we encourage users to use hdf5 format.
```
python data.py
```

## Config
For custom hyperparameters and configurations, modify:

`electric-field-fno/exp-l2.yaml`

## Run
```
python main.py --config-name {config-name}.yaml
```
Sweep over hparams :
```
python main.py --config-name {config-name}.yaml --multirun seed=7,42,170 training-data-dir={path-to-data} test-data-dir={path-to-data}
```

**Be sure** to fill experiment name at `experiement:` field.

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
This will drop a folder with experiment name and datetime under `runs`.

In the folder, **.hydra** folder, **tensorboard** event file and **model** checkpoint will be saved.