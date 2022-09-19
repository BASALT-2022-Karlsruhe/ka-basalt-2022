import os


def set_env(eval_episodes=10, eval_max_steps=1000):
    os.environ["AICROWD_NUM_EVAL_EPISODES"] = eval_episodes
    os.environ["AICROWD_NUM_EVAL_MAX_STEPS"] = eval_max_steps