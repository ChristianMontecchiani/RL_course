import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["SDL_VIDEODRIVER"] = "dummy" # for pygame rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb

import reacher
from agent import Agent, Policy
from common import helper as h
from common import logger as logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Policy training function
def train(agent, env, min_update_samples=2000):
    # Run actual training        
    reward_sum, timesteps, num_updates = 0, 0, 0
    done = False

    # Reset the environment and observe the initial state
    observation = env.reset()

    while not done:
        # Get action from the agent
        action, action_log_prob = agent.get_action(observation)
        previous_observation = observation.copy()

        # Perform the action on the environment, get new state and reward
        observation, reward, done, _ = env.step(action)

        # Store action's outcome (so that the agent can improve its policy)
        agent.store_outcome(previous_observation, action, observation,
                reward, action_log_prob, done)

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # Update the policy, if we have enough data
        if len(agent.states) > min_update_samples:
            agent.update_policy()
            num_updates += 1

    # Return stats of training
    update_info = {'timesteps': timesteps,
            'ep_reward': reward_sum,
            'num_updates': num_updates}
    return update_info


# Function to test a trained policy
def test(agent, env, episodes):
    total_test_reward, total_test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()

        test_reward, test_len = 0, 0
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)

            test_reward += reward
            test_len += 1
        total_test_reward += test_reward
        total_test_len += test_len
        print("Test ep reward:", test_reward)
    print("Average test reward:", total_test_reward/episodes, "episode length:", total_test_len/episodes)


# The main function
@hydra.main(config_path='cfg', config_name='ex1_cfg')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    
    run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # use wandb to store stats; we aren't currently logging anything into wandb during testing (might be useful to
    # have the cfg.testing check here if someone forgets to set use_wandb=false)
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # create env
    # TODO: Task 1: Train with 100 steps, test with 1000 steps
    env = gym.make(cfg.env_name, 
                    max_episode_steps=cfg.max_episode_steps,
                    render_mode='rgb_array')
    # seeding the environemnt
    env.seed(cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/cfg.env_name/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/cfg.env_name/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode

    
    # Get dimensionalities of actions and observations
    action_space_dim = h.get_space_dim(env.action_space)
    observation_space_dim = h.get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, cfg.batch_size)

    # Print some stuff
    print("Environment:", cfg.env_name)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if not cfg.testing: # training
        for ep in range(cfg.train_episodes+1):
            train_info = train(agent, env,
                                min_update_samples=cfg.min_update_samples)
            train_info.update({'episodes': ep})

            if not cfg.silent:
                print(f"Episode {ep} finished. Total reward: {train_info['ep_reward']} ({train_info['timesteps']} timesteps)")
            
            if cfg.use_wandb: 
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)

        # Save the model
        if cfg.save_model:
            model_path = work_dir/'model'/f'{cfg.env_name}_rew1_params.pt'
            torch.save(policy.state_dict(), model_path)
            print("Model saved to", model_path)
        
        if cfg.save_logging:
            logging_path = work_dir/'logging'/f'{cfg.env_name}_logging.pkl'
            L.save(logging_path)

        print("------Training finished.------")

    else: # testing
        if cfg.model_path == 'default':
            cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'
        print("Loading model from", cfg.model_path, "...")
        
        # load model
        state_dict = torch.load(cfg.model_path)
        policy.load_state_dict(state_dict)
        
        print("Testing...")
        test(agent, env, 10)


# Entry point of the script
if __name__ == "__main__":
    main()


