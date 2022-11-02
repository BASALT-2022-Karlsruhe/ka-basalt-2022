if __name__ == "__main__":
    from run_agent2 import main

    model = "data/VPT-models/foundation-model-1x.model"
    weights = "data/VPT-models/foundation-model-1x.weights"
    env = "MineRLBasaltFindCave-v0"
    show = False
    record = False
    max_steps = 1000
    video_dir = "./video"
    main(
        model=model,
        weights=weights,
        env=env,
        show=show,
        record=record,
        max_steps=max_steps,
        video_dir=video_dir,
    )
