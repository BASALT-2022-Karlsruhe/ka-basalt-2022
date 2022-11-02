import logging
import pickle

import aicrowd_gym
import coloredlogs
import minerl

from config import EVAL_EPISODES, EVAL_MAX_STEPS
from openai_vpt.agent import MineRLAgent
from train import FOUNDATION_MODEL_FILE, SUBMISSION_WEIGHTS_PATH

coloredlogs.install(logging.DEBUG)


ENV_STRING = "BuildVillageHouse"
MINERL_GYM_ENV = f'MineRLBasalt{ENV_STRING}-v0'


def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    model = f"data/VPT-models/{FOUNDATION_MODEL_FILE}"
    weights = SUBMISSION_WEIGHTS_PATH.format(ENV_STRING)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for i in range(EVAL_EPISODES):
        agent.reset()
        obs = env.reset()
        done = False
        for step_counter in range(EVAL_MAX_STEPS):
            
            action = agent.get_action(obs)
            action["ESC"] = 0

            obs, reward, done, info = env.step(action)

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == '__main__':
    main()
