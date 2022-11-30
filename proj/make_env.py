import gym
import yaml


GYM_TASKS = {
    'MountainCarContinuous-v0',
    'LunarLander-v2',
    'BipedalWalker-v3',
    'Hopper-v2',
    'Hopper-v4'
}

ALGOS = {
    "Actor_Critic", 
    "Reinforce", 
    "DDPG", 
    "DQN", 
    "AlphaZero", 
    "PETS", 
}

def create_env(config_file_name, seed):
    config = yaml.load(open(f'./cfg/{config_file_name}.yaml', 'r'),  Loader=yaml.Loader)

    if config['env_name'] in GYM_TASKS:
        env_kwargs = config['env_parameters']
        if env_kwargs is None:
            env_kwargs = dict()
        env = gym.make(config['env_name'], **env_kwargs)
        env.reset(seed=seed)

    else:
        raise NameError("Wrong environment name was provided in the config file! Please report this to @Nikita Kostin.")

    return env
