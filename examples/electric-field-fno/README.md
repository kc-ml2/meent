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

## Electric field prediction
We provide the pretrained model weights and demonstration of the model.

Please refer to `evaluation.ipynb`.

## Generate Dataset
For this experiment, we stick to simple pickle format, but for larger, complex datasets, we encourage users to use hdf5 format.
```
python data.py
```
## Training
### Config
For custom hyperparameters and configurations, modify:

`electric-field-fno/exp-l2.yaml`

```
experiment: tfno-l2
device: cuda:0
################################################################################
seed : 42 #7 # 42, 170
model: CarstenNO
model_config:
  n_modes: [24, 24]
  in_channels: 1
  out_channels: 2
  hidden_channels: 32
...
```

### Run
You can run training process with below command. You must pass data directories with the data generated from `data.py` or you own data generation script.
```
python main.py --config-name {config-name}.yaml --train_data_dir {path-to-data} --test_data_dir {path-to-data}
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