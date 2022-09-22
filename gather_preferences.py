# TODO
# - load trajectory pairs from given path
# - present trajectories side by side to the human
# - let human input their preference (a single number indicating which trajectory is preferred OR indifference i.e. 0.5)
# - save preferences alongside the trajectory pairs


def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=False, record=False, video_dir="./video"):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)

    env._max_episode_steps = max_steps
    if record:
        # enable recording
        env.metadata['render.modes'] = ["rgb_array", "ansi"]
        env = Monitor(env, video_dir, video_callable=lambda episode_id: True, force=True)

    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for _ in range(n_episodes):
        obs = env.reset()
        while True:
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if done:
                break
    env.close()