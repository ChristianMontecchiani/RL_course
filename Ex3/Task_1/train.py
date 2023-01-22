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


def init_q_table(observation_space, action_dim, discr, bool_position=None, init_q=0.):
    """ Return a q_table, which shape [*state.shape, action_dim] and the corresponding axis."""
    high_values = observation_space.high
    low_values = observation_space.low
    axis = []
    for idx, (low_val, high_val) in enumerate(zip(low_values, high_values)):
        # here to avoid inf boundary, we truncate the value to [-4, 4] 
        if low_val < -1e10: low_val = -4
        if high_val > 1e10: high_val = 4

        if (bool_position is not None) and (idx in bool_position):
            axis.append(np.linspace(low_val, high_val, 2, dtype=np.float32)) # for boolean, we only have two values: 1., 0.
        else:
            axis.append(np.linspace(low_val, high_val, discr, dtype=np.float32))

    _shape = [ax.shape[0] for ax in axis] + [action_dim]
    q_table = np.zeros(_shape) + init_q

    #print(q_table.shape)
    #print(f"Axis shape: {axis[0].shape}")
    return axis, q_table


def get_table_idx(state, axis):
    """ Give a state, discrete it and return the index in each dimension (axis). 
    With the returned index, you can access q(s,.) with q_table[idx]."""
    def _get_ax_idx(ax, value):
        return np.argmin(np.abs(ax - value))
    return tuple([_get_ax_idx(ax, value) for ax, value in zip(axis, state)])


def get_action(state, q_axis, q_table, epsilon=0.0):
    # if epsilon == 0.0, the policy will be greedy -- always choose the best action

    # TODO: Implement epsilon-greedy
    ########## Your code starts here ##########

    q_value = q_table[get_table_idx(state, q_axis)]
   # print(q_value)
    actions_len = len(q_value)

    action = np.argmax(q_value)
    u = np.random.random()
    if u <= epsilon: 
        action = np.random.randint(actions_len)

    return action
    ########## Your code ends here #########


def update_q_value(old_state, action, new_state, gamma, reward, done, alpha, q_axis, q_table):
    # TODO: Task 1.1, update q value
    
    old_table_idx = get_table_idx(old_state, q_axis) # idx of q(s_old, *)
    new_table_idx = get_table_idx(new_state, q_axis) # idx of q(s_new, *)   
    ########## Your code starts here ##########

    if not done:
        old_q_value = q_table[old_table_idx][action]
        max_q_value = np.max(q_table[new_table_idx])

        update = old_q_value + alpha*(reward + gamma*max_q_value - old_q_value)
        q_table[old_table_idx][action] = update 
    ########### Your code ends here ##########
    return q_table



@hydra.main(config_path='cfg', config_name='ex3_cfg')
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_logging: 
        L = logger.Logger() # create a simple logger to record stats

    # use wandb to store stats
    if cfg.use_wandb:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{run_id}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # create env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)
    env.seed(cfg.seed)
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir/'video'/'train',
                                        episode_trigger=lambda x: (x % 500 == 0),
                                        name_prefix=cfg.exp_name) # save video for testing every 5 episodes

    # init q_table with zeros
    q_axis, q_table = init_q_table(env.observation_space, 
        env.action_space.n, cfg.discr, bool_position=cfg.bool_position, init_q=cfg.initial_q)

    # begin training and testing
    ep_reward_deque = deque([], maxlen=500)  # used to calculate the smooted (avg over recent 500 episodes) ep_reward
    for ep in range(cfg.train_episodes + 1):
        # set epsilon value
        if cfg.epsilon == 'glie':
            epsilon = cfg.glie_b / (cfg.glie_b + ep)  
        elif isinstance(cfg.epsilon, (int, float)):
            epsilon = cfg.epsilon
        else: 
            raise ValueError

        state, done, ep_reward, timesteps = env.reset(), False, 0, 0
        while not done:
            action = get_action(state, q_axis, q_table, epsilon=epsilon) 
            new_state, reward, done, _ = env.step(action)

            q_table = update_q_value(state, action, new_state, cfg.gamma, reward, done, cfg.alpha,
                                        q_axis, q_table)
    
            state = new_state
            ep_reward += reward
            timesteps += 1
        
        ep_reward_deque.append(ep_reward)
        info = {
            'episode': ep,
            'epsilon': epsilon,
            'ep_reward': ep_reward,
            'timesteps': timesteps,
            'ep_reward_avg': np.mean(list(ep_reward_deque))
        }

        if cfg.use_wandb: wandb.log(info)
        if cfg.save_logging: L.log(**info)

        if (not cfg.silent) and (ep % 500 == 0): print(info)

    # save the q-value table and q_axis
    h.save_object({'q_table': q_table, 'axis': q_axis},
                        work_dir/ (cfg.fn_q_table + ".pkl"))

    if cfg.save_logging:
        logging_path = work_dir/(cfg.fn_logging + '.pkl')
        L.save(logging_path)


if __name__ == "__main__":
    main()