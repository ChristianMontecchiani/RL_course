import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["SDL_VIDEODRIVER"] = "dummy" # for pygame rendering
import time
from pathlib import Path
from collections import deque

import gym
import numpy as np
import hydra
import wandb

from common import helper as h
from common import logger as logger

from train import get_action


@hydra.main(config_path='cfg', config_name='ex3_cfg')
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'

    # create env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)
    env.seed(cfg.seed)

    # record video if needed
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir / 'video' / 'test',
                                       episode_trigger=lambda x: x % 1 == 0,
                                       name_prefix=cfg.exp_name)

    # load q_table
    data = h.load_object(work_dir / cfg.fn_q_table + ".pkl")
    q_axis, q_table = data['axis'], data['q_table']

    # begin testing
    for ep in range(cfg.test_episodes):
        state, done, ep_reward, timesteps = env.reset(), False, 0, 0
        while not done:
            action = get_action(state, q_axis, q_table, epsilon=0.0) # be greedy during testing
            new_state, reward, done, _ = env.step(action)

            state = new_state
            ep_reward += reward
            timesteps += 1
        
        info = {
            'test_episode': ep,
            'test_ep_reward': ep_reward,
            'timesteps': timesteps,
        }

        if not cfg.silent: print(info)


if __name__ == "__main__":
    main()