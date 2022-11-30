import numpy as np

def evaluate(env, agent, env_step=0, num_episode=10, video_recorder=None):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episode):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video_recorder: video_recorder.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, t0=(t==0))

			obs, reward, done, _ = env.step(action.cpu().numpy())

			ep_reward += reward
			if video_recorder: video_recorder.record(env)
			t += 1
		episode_rewards.append(ep_reward)

		if video_recorder: 
			video_recorder.save(env_step)
            
	return episode_rewards