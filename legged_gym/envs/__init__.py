from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.go2.go2_env import Go2Robot

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())

from .go2_him.go2_him_config import Go2HIMCfg, Go2HIMCfgPPO
from .go2_him.go2_him_env import Go2HIMRobot

task_registry.register("go2_him", Go2HIMRobot, Go2HIMCfg(), Go2HIMCfgPPO())

from .go2_him.retrain_config import Go2HIMRetrainCfg, Go2HIMRetrainCfgPPO
from .go2_him.retrain_env import Go2HIMRetrainRobot
task_registry.register("go2_him_retrain", Go2HIMRetrainRobot, Go2HIMRetrainCfg(), Go2HIMRetrainCfgPPO())

from .go2_him.upstream_flat_config import Go2HIMUpstreamFlatCfg, Go2HIMUpstreamFlatCfgPPO
task_registry.register('go2_him_upstream_flat', Go2HIMRobot, Go2HIMUpstreamFlatCfg(), Go2HIMUpstreamFlatCfgPPO())

from .go2_him.candidate_config import (Go2HIMCandidateFlatCfg, Go2HIMCandidateFlatCfgPPO,
                                      Go2HIMCandidateRoughCfg, Go2HIMCandidateRoughCfgPPO)
task_registry.register('go2_him_candidate_flat', Go2HIMRetrainRobot, Go2HIMCandidateFlatCfg(), Go2HIMCandidateFlatCfgPPO())
task_registry.register('go2_him_candidate_rough', Go2HIMRetrainRobot, Go2HIMCandidateRoughCfg(), Go2HIMCandidateRoughCfgPPO())
