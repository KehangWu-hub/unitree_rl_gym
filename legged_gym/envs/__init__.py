from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.go2.go2_env import Go2Robot

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())

from .go2_him.go2_him_config import Go2HIMCfg, Go2HIMCfgPPO
from .go2_him.go2_him_env import Go2HIMRobot

task_registry.register("go2_him", Go2HIMRobot, Go2HIMCfg(), Go2HIMCfgPPO())
