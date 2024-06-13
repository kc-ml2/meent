from ray.rllib.algorithms.simple_q import SimpleQConfig


"""
config for reproducing the same result as original code
"""
single_worker = SimpleQConfig()
single_worker.training(
    model={
        'no_final_linear': True,
        'vf_share_layers': False,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "ReplayBuffer",
        # "type": "MultiAgentReplayBuffer", # when num_workers > 0
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    # dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=512,
    tau=0.1,
).resources(
    num_gpus=1
).rollouts(
    horizon=128,
    num_rollout_workers=0, # important!! each accounts for process
    num_envs_per_worker=1, # each accounts for process
    rollout_fragment_length=2,
).exploration(
    explore=True,
    exploration_config={
        "type": "EpsilonGreedy",
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 100000,
    }
)

"""
config for parallelizing the original code
"""
multiple_worker = SimpleQConfig()
multiple_worker.training(
    model={
        'no_final_linear': True,
        'vf_share_layers': False,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "ReplayBuffer",
        # "type": "MultiAgentReplayBuffer", # when num_workers > 0
        "learning_starts": 1000,
        "capacity": 25000,
        "replay_sequence_length": 1,
    },
    # dueling=False,
    lr=0.001,
    gamma=0.99,
    train_batch_size=256,
    tau=0.1,
).resources(
    num_gpus=2
).rollouts(
    horizon=512,
    num_rollout_workers=16,
    num_envs_per_worker=1,
    rollout_fragment_length=2,
).exploration(
    explore=True,
    exploration_config={
        "type": "EpsilonGreedy",
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 50000,
    }
)