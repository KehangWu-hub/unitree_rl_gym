"""Selective training changes, with matched flat and mixed-terrain candidates."""
from .go2_him_config import Go2HIMCfgPPO
from .retrain_config import Go2HIMRetrainCfg, Go2HIMRetrainCfgPPO


class Go2HIMCandidateFlatCfg(Go2HIMRetrainCfg):
    training_profile = 'go2_him_candidate_flat_v1'
    training_task = 'go2_him_candidate_flat'
    action_distribution = 'gaussian'
    hardware_deployment_blocked = True

    class env(Go2HIMRetrainCfg.env):
        # Full mixed mesh + 100-step PPO exceeded 6GB at 2048 environments.
        num_envs = 1024

    class normalization(Go2HIMRetrainCfg.normalization):
        clip_actions = 100.


class Go2HIMCandidateRoughCfg(Go2HIMCandidateFlatCfg):
    training_profile = 'go2_him_candidate_rough_v1'
    training_task = 'go2_him_candidate_rough'

    class terrain(Go2HIMCandidateFlatCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        # Same five terrain buckets as /home/wkh/projects/legged_gym.
        terrain_proportions = [.1, .1, .35, .25, .2]
        max_init_terrain_level = 1


class Go2HIMCandidateFlatCfgPPO(Go2HIMRetrainCfgPPO):
    class policy(Go2HIMCfgPPO.policy):
        init_noise_std = .6

    class algorithm(Go2HIMRetrainCfgPPO.algorithm):
        schedule = 'adaptive'

    class runner(Go2HIMRetrainCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        experiment_name = 'go2_him_candidate_flat'
        num_steps_per_env = 100
        # Same 131072000 transition budget as v2's 1024 * 32 * 4000.
        max_iterations = 1280
        save_interval = 100


class Go2HIMCandidateRoughCfgPPO(Go2HIMCandidateFlatCfgPPO):
    class runner(Go2HIMCandidateFlatCfgPPO.runner):
        experiment_name = 'go2_him_candidate_rough'
