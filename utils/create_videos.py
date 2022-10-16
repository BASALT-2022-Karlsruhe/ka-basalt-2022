from run_agent import main as run_agent_main
from pathlib import Path


def create_videos(env, foundation_model, eval_episodes, eval_max_steps, show=True):
    video_dir = Path(f"./train/videos/{env}")
    video_dir.mkdir(parents=True, exist_ok=True)
    run_agent_main(
        model=f"data/VPT-models/{foundation_model}.model",
        weights=f"train/MineRLBasalt{env}.weights",
        env=f"MineRLBasalt{env}-v0",
        n_episodes=eval_episodes,
        max_steps=eval_max_steps,
        video_dir=str(video_dir.absolute()),
        show=show,
        record=True
    )
