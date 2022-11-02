import json
import os
import pickle
import uuid
from argparse import ArgumentParser

import aicrowd_gym
import minerl  # noqa
from gym.wrappers import Monitor

from data_loader import env_action_to_json_action
from openai_vpt.agent import MineRLAgent


def write_jsonl(action_dicts, file_path):
    """Writes actions as JSON objects into .jsonl file"""
    with open(file_path + ".jsonl", "w") as file:
        for action in action_dicts:
            json.dump(action, file)
            file.write("\n")


def generate_trajectories(
    model,
    weights,
    env_string,
    n_episodes=1,
    max_steps=int(1e9),
    show=False,
    video_dir="./video",
    verbose=1
):
    """Generate video and action files for a number of episodes sampled using the give model."""
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make("MineRLBasalt" + env_string + "-v0")
    env._max_episode_steps = max_steps

    # Enable recording via Monitor wrapper
    env.metadata["render.modes"] = ["rgb_array", "ansi"]
    unique_id = uuid.uuid4()
    env = Monitor(
        env,
        video_dir,
        video_callable=lambda episode_id: True,
        force=False,
        resume=True,
        uid=unique_id,
    )

    # Build agent and load weights
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for ep in range(n_episodes):
        file_name = f"{env.file_prefix}.video.{env.file_infix}.video{env.episode_id:06}"
        if verbose > 0:
            print(f"Generating trajectory {ep + 1} / {n_episodes}: {file_name}")
        agent.reset()
        obs = env.reset()
        action_dicts = []
        prev_json_action = None
        step = 0
        while True:
            # Build agent action
            action = agent.get_action(obs)
            action["ESC"] = 0

            # Record action as JSON compatible dict
            action_dict = env_action_to_json_action(action, prev_json_action)
            action_dicts.append(action_dict)
            prev_json_action = action_dict

            obs, _, done, _ = env.step(action)

            if verbose > 0 and step % 100 == 99:
                print(f"Step {step}")
            step += 1

            if show:
                env.render()
            if done:
                break
        # Save actions as JSONL file
        write_jsonl(action_dicts, os.path.join(video_dir, file_name))
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to the '.model' file to be loaded.",
        default="data/VPT-models/foundation-model-1x.model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to the '.weights' file to be loaded.",
        default="data/VPT-models/foundation-model-1x.weights",
    )
    parser.add_argument("--env", type=str, default="MineRLBasaltFindCave-v0")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--video_dir", type=str, default="./video")

    args = parser.parse_args()

    generate_trajectories(
        args.model,
        args.weights,
        args.env,
        n_episodes=args.n_episodes,
        show=args.show,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
    )
