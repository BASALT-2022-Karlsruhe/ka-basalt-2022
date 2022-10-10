import gym
from gym.envs.registration import register

from gym_wrappers import (
    DictToMultiDiscreteActionSpace,
    HiddenStateObservationSpace,
    ObservationToCPU,
)


def sb3_minerl_findcave_env(minerl_agent):
    env = gym.make("MineRLBasaltFindCave-v0")

    # Make env compatible with SB3
    sb3_env = DictToMultiDiscreteActionSpace(env, minerl_agent)
    sb3_env = HiddenStateObservationSpace(sb3_env, minerl_agent)
    sb3_env = ObservationToCPU(sb3_env)

    return sb3_env


def sb3_minerl_makewaterfall_env(minerl_agent):
    env = gym.make("MineRLBasaltMakeWaterfall-v0")

    # Make env compatible with SB3
    sb3_env = DictToMultiDiscreteActionSpace(env, minerl_agent)
    sb3_env = HiddenStateObservationSpace(sb3_env, minerl_agent)
    sb3_env = ObservationToCPU(sb3_env)

    return sb3_env


def sb3_minerl_buildvillagehouse_env(minerl_agent):
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")

    # Make env compatible with SB3
    sb3_env = DictToMultiDiscreteActionSpace(env, minerl_agent)
    sb3_env = HiddenStateObservationSpace(sb3_env, minerl_agent)
    sb3_env = ObservationToCPU(sb3_env)

    return sb3_env


def sb3_minerl_createvillageanimalpen_env(minerl_agent):
    env = gym.make("MineRLBasaltCreateVillageAnimalPen-v0")

    # Make env compatible with SB3
    sb3_env = DictToMultiDiscreteActionSpace(env, minerl_agent)
    sb3_env = HiddenStateObservationSpace(sb3_env, minerl_agent)
    sb3_env = ObservationToCPU(sb3_env)

    return sb3_env


for env_name in [
    "FindCave",
    "MakeWaterfall",
    "BuildVillageHouse",
    "CreateVillageAnimalPen",
]:
    register(
        "MineRLBasalt" + env_name + "SB3" + "-v0",
        entry_point=f"sb3_minerl_envs:sb3_minerl_{env_name.lower()}_env",
    )
