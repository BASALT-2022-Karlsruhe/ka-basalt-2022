# TODO Generate Trajectory Pairs
# - run rollouts for each task and record video and input data
# - preprocess video and input stream into observation and action data
# - select trajectory pairs among the produced rollouts based on some criterion (e.g. random choice, differences in trajectories, variance in an ensemble of reward models)
# - safe the preprocessed trajectory pairs

# TODO Gather Preferences
# - load trajectory pairs from given path
# - present trajectories side by side to the human
# - let human input their preference (a single number indicating which trajectory is preferred OR indifference i.e. 0.5)
# - save preferences alongside the trajectory pairs

# TODO Train Reward Model
# - load trajectory pairs and gathered preferences from given path
# - load or initialize reward model
# - train the reward model with minibatch stochastic gradient descent, for each minibatch
#   - feed observation-action-pairs through the reward model to receive a reward trajectory
#   - use the Bradly-Terry model to compute the estimated preferences
#   - compute loss using the gathered preferences
#   - do the backward pass to compute gradients
#   - take a gradient step
# - save the updated reward model

# TODO Finetune Policy using RL
# - create MineRLAgent from existing BC pretrained model
# - load MineRLRewardNet trained with human feedback
# - use StableBaselines3 to finetune the agent under the reward model
# - save the resulting MineRLAgent