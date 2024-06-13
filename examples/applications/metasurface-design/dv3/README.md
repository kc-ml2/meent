# Metasurface design with DreamerV3

We adapted code from [SheepRL](https://github.com/Eclectic-Sheep/sheeprl), and the process is controlled by [hydra](https://hydra.cc/docs/intro/).



## Setup
```
pip install -e .
```


## Dynamics prediction
We provide the pretrained model weights and demonstration of the model.

Please refer to `dynamics-prediction.ipynb`.

## Training
### Config
For custom hyperparameters and configurations, modify:
`sheeprl/configs/exp/dreamer_v3_metasurface.yaml`
```
seed: 42

# Algorithm
algo:
  replay_ratio: 2
  total_steps: 50000
  horizon: 15
  per_rank_batch_size: 8
  per_rank_sequence_length: 32
...
```

### Run
Must pass experiment name `exp`, specified by the name of yaml file.
You can pass additional arguments, like `seed` in below commandline.
```
python sheeprl.py exp=dreamer_v3_metasurface seed={your-seed}
```