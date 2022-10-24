from argparse import ArgumentParser
import pickle
import queue

import aicrowd_gym
import minerl

from gym.wrappers import Monitor

from openai_vpt.agent import MineRLAgent
from find_cave_classifier import FindCaveCNN, preprocessing

ESC_MODELS = {"MineRLBasaltFindCave": FindCaveCNN}


def main(model, weights, esc_model_path, env, n_episodes=3, max_steps=int(1e9), show=False, record=False, video_dir="./video"):
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

    esc_model = ESC_MODELS[env]()
    esc_model.load_state_dict(esc_model_path)

    for _ in range(n_episodes):
        obs = env.reset()
        n_steps = 0

        obs_queue = queue.Queue(maxsize=4)
        while True:
            obs_queue.put(preprocessing(obs["pov"]))
            action = agent.get_action(obs)

            if n_steps < 4:
                action["ESC"] = 0
            else:
                obs_stack = th.from_numpy(np.array(list(obs_queue)))
                esc_action = esc_model.predict(obs_stack)
                action["ESC"] = esc_action

            obs, _, done, _ = env.step(action)
            if show:
                if n_steps % 10 == 0:
                    print(f"Step {n_steps}")
                env.render()
            n_steps += 1
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max_steps", type=int, required=False, default=1000)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--record", action="store_true", help="Record the rendered environment.")
    parser.add_argument("--video_dir", type=str, required=False, default="./video")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, record=args.record, max_steps=args.max_steps, video_dir=args.video_dir)
