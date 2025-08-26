import gym
from crafter import crafter
from utils import env_wrapper
from model import CustomResNet, CustomACPolicy, CustomPPO, TQDMProgressBar
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from gym.wrappers import FrameStack
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold


def make_env(config, mode):
    env = gym.make("MyCrafter-v0")
    env = env_wrapper.MineStoneWrapper2(env, decay_steps=0)
    if mode == "train" and config["recorder"]:
        env = crafter.Recorder(
            env, config["recorder_res_path"],
            save_stats = True,
            save_video = False,
            save_episode = False,
        )
    env = env_wrapper.InitWrapper(env, init_items=config["init_items"], init_num=config["init_num"])
    return env


if __name__ == "__main__":

    config = {
        "total_timesteps": 2000000,
        "save": True,
        "save_dir": "./stone",
        "init_items": ["wood_pickaxe"],
        "init_num": [1],
        "recorder": False,
        "recorder_res_path": "comparisons/res/stone",
        "early_stop": False,
        "stop_threshold": 9000,
    }

    train_env = make_env(config, mode="train")
    eval_env = make_env(config, mode="eval")

    policy_kwargs = {
        "features_extractor_class": CustomResNet,
        "features_extractor_kwargs": {"features_dim": 1024},
        "activation_fn": nn.ReLU,
        "net_arch": [],
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5}
    }

    model = CustomPPO(
        CustomACPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        # learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.65,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=False
    )

    total_timesteps = config["total_timesteps"]

    call_back_on_best = StopTrainingOnRewardThreshold(reward_threshold=config["stop_threshold"], verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=call_back_on_best,
        eval_freq=10000,
        verbose=1,
        n_eval_episodes=10,
        deterministic=False,
    )

    if config["early_stop"]:
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

    if config["save"]:
        model.save(config["save_dir"])

    train_env.close()
    eval_env.close()
