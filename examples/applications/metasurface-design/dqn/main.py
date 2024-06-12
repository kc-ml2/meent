import argparse
from datetime import datetime
from pathlib import Path
import os
from operator import itemgetter

from tqdm import tqdm
import numpy as np

import torch

import ray
from ray import air, tune
from ray.tune import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog

from gym.wrappers import TimeLimit
import deflector_gym
from deflector_gym.wrappers import BestRecorder, ExpandObservation

from model import ShallowUQNet
from utils import StructureWriter, seed_all

DATA_DIR = None
PRETRAINED_CKPT = None
LOG_DIR = None

"""
seeding needs to be taken care when multiple workers are used,
that is, you need to set seed for each worker
"""
SEED = 42
# seed_all(SEED)

def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )
class Callbacks(DefaultCallbacks):
    """
    logging class for rllib
    the method's name itself stands for the logging timing

    e.g. 
    `if step % train_interval == 0:` is equivalent to `on_learn_on_batch`

    you may feel uncomfortable with this logging procedure, 
    but when distributed training is used, it's inevitable
    """

    def _get_max(self, base_env):
        # retrieve `env.best`, where env is wrapped with BestWrapper to record the best structure
        bests = [e.best for e in base_env.get_sub_environments()]
        best = max(bests, key=itemgetter(0))

        return best[0], best[1]

    def _tb_image(self, structure):
        # transform structure to tensorboard addable image
        img = structure[np.newaxis, np.newaxis, :].repeat(32, axis=1)

        return img

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs) -> None:
        eff, struct = self._get_max(base_env)

        episode.custom_metrics['initial_efficiency'] = eff

    def _j(self, a, b):
        return os.path.join(a, b)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs, ) -> None:
        eff, struct = self._get_max(base_env)
        episode.custom_metrics['max_efficiency'] = eff
        # filename = 'w' + str(worker.worker_index) + f'_{eff * 100:.6f}'.replace('.', '-')
        # filename = self._j(LOG_DIR, filename)
        # np.save(filename, struct)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', type=bool, default=True,
        help='True for train mode, False for test mode'
    )
    
    parser.add_argument(
        '--data_dir', type=str, default='final',
        help='absolute path to data directory'
    )
    parser.add_argument(
        '--wavelength', type=int, default=1100,
        help='wavelength of the incident light'
    )
    parser.add_argument(
        '--angle', type=int, default=70,
        help='target deflection angle condition'
    )
    parser.add_argument(
        '--thickness', type=int, default=325,
        help='thickness of the pillar'
    )
    parser.add_argument(
        '--train_steps', type=int, default=50000,
        help='number of training steps'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed'
    )
    parser.add_argument(
        '--num_cpus_per_worker', type=int, default=4,
        help='random seed'
    )
    parser.add_argument(
        '--num_rollout_workers', type=int, default=8,
        help='random seed'
    )

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    LOG_DIR = f"{args.data_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    PRETRAINED_CKPT = args.pretrained_ckpt
    SEED = args.seed

    os.makedirs(LOG_DIR, exist_ok=True)


    env_id = 'MeentIndex-v0'
    env_config = {'wavelength': args.wavelength, 'desired_angle': args.angle, 'thickness': args.thickness}#, 'seed': SEED}
    model_cls = ShallowUQNet

    def make_env(config):
        # config['seed'] = config['seed'] + config.worker_index
        # print(f"set worker {config.worker_index}'s seed to {config.seed}" )
        env = deflector_gym.make(env_id, **config)
        env = BestRecorder(env)
        env = ExpandObservation(env)
        # env = StructureWriter(env, DATA_DIR)
        env = TimeLimit(env, max_episode_steps=128)

        return env


    from configs.simple_q import multiple_worker as config
    ray.init(
        local_mode=False,# num_cpus=80, num_gpus=1, 
    )

    register_env(env_id, lambda c: make_env(env_config))
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)
    config.resources(num_cpus_per_worker=args.num_cpus_per_worker, num_gpus=1)
    config.rollouts(num_rollout_workers=args.num_rollout_workers)
    
    config.policies = None
    config.framework(
        framework='torch'
    ).environment(
        env=env_id,
        env_config=env_config,
        normalize_actions=False
    ).callbacks(
        Callbacks  # register logging
    ).training(
        model={'custom_model': model_cls}
    ).debugging(
        seed=SEED
        #log_level='DEBUG'
        # seed=tune.grid_search([1, 2, 3, 4, 5]) # if you want to run experiments with multiple seeds
    )

    algo = config.build()
    print('*'*20, count_model_params(algo.get_policy().model))
    
    stop = {
        "timesteps_total": args.train_steps,
    }
    tuner = tune.Tuner(
        'SimpleQ',
        param_space=config.to_dict(),
        # tune_config=tune.TuneConfig(), # for hparam search
        run_config=air.RunConfig(
            stop=stop,
            local_dir=DATA_DIR,
            # name=LOG_DIR,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute='episode_reward_max',
                checkpoint_score_order='max',
                checkpoint_frequency=50,
                checkpoint_at_end=True,
            ),
        ),
    )

    results = tuner.fit()



