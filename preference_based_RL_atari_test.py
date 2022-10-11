from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import preference_comparisons
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import (
    CnnRewardNet,
    NormalizedRewardNet,
)
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from gym_wrappers import ObservationToInfos


def main():

    venv = make_vec_env(
        "Pong-v0",
        post_wrappers=[
            lambda env, i: AtariWrapper(ObservationToInfos(env)),
        ],
    )

    reward_net = CnnRewardNet(
        venv.observation_space,
        venv.action_space,
        use_action=True,
    )
    normalized_reward_net = NormalizedRewardNet(reward_net, RunningNorm)
    preference_model = preference_comparisons.PreferenceModel(normalized_reward_net)

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)

    # gatherer = preference_comparisons.SyntheticGatherer(seed=0)
    gatherer = preference_comparisons.PrefCollectGatherer(
        pref_collect_address="http://127.0.0.1:8000",
        video_output_dir="/home/robert/Projects/BASALT/pref_collect/videofiles/",
    )

    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=normalized_reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=3,
    )

    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        seed=0,
        n_steps=512 // venv.num_envs,
        batch_size=32,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=0,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=10,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=True,
        seed=0,
        initial_epoch_multiplier=1,
    )

    # Run training
    pref_comparisons.train(
        total_timesteps=15_000,  # For good performance this should be 1_000_000
        total_comparisons=100,  # For good performance this should be 5_000
    )

    reward, _ = evaluate_policy(agent.policy, venv, 10)

    print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
