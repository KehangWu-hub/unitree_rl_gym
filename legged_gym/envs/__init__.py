from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.go2.go2_env import Go2Robot

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())
