# Metasurface design with DreamerV3

We adapted code from [SheepRL](https://github.com/Eclectic-Sheep/sheeprl), and the process is controlled by [hydra](https://hydra.cc/docs/intro/).



## Setup
```
pip install -e .
```

## Config
For custom hyperparameters and configurations, modify:
`sheeprl/configs/exp/dreamer_v3_metasurface.yaml`


## Run
Must specify experiment name `exp`, specified by the name of yaml file.
You can pass additional arguments, like `seed` in below commandline.
```
python sheeprl.py exp=dreamer_v3_metasurface seed={SEED}
```

## Dynamics prediction
Please refer to `dynamics-prediction.ipynb`.