from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS
from pathlib import Path

def main():
    video_dir = Path("./video/find_cave").mkdir(parents=True,exist_ok=True)
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        weights="train/MineRLBasaltFindCave.weights",
        env="MineRLBasaltFindCave-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
        video_dir=str(video_dir)
    )

if __name__ == '__main__':
    main()