"""From-scratch flat-ground HIM profile using direct policy joint targets."""
from .go2_him_config import Go2HIMCfg, Go2HIMCfgPPO


class Go2HIMRetrainCfg(Go2HIMCfg):
    training_profile = 'go2_him_direct_targets_v3'

    class env(Go2HIMCfg.env):
        num_envs = 1024
        episode_length_s = 20.

    class terrain(Go2HIMCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True

    class commands(Go2HIMCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 4.
        class ranges(Go2HIMCfg.commands.ranges):
            lin_vel_x = [-.5, .5]
            lin_vel_y = [-.35, .35]
            ang_vel_yaw = [-.5, .5]

    class control(Go2HIMCfg.control):
        decimation = 10

    class normalization(Go2HIMCfg.normalization):
        clip_actions = 4.

    class domain_rand(Go2HIMCfg.domain_rand):
        payload_mass_range = [-1., 2.]
        com_displacement_range = [-.025, .025]
        friction_range = [.5, 1.25]
        initial_joint_pos_range = [.9, 1.1]
        push_interval_s = 12.
        max_push_vel_xy = .3
        disturbance_range = [-15., 15.]

    class rewards(Go2HIMCfg.rewards):
        tracking_sigma = .05
        base_height_target = .30
        class scales(Go2HIMCfg.rewards.scales):
            termination = -5.
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.
            base_height = -1.
            orientation = -1.
            foot_clearance = -.01
            action_rate = -.01
            smoothness = -.01
            collision = -.5
            upright_height = .5
            idle_motion = -.02
            feet_slip = -.1

    class sim(Go2HIMCfg.sim):
        dt = .002


class Go2HIMRetrainCfgPPO(Go2HIMCfgPPO):
    seed = 42
    class policy(Go2HIMCfgPPO.policy):
        init_noise_std = .6
        action_bound = 4.
    class algorithm(Go2HIMCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = 'fixed'
        entropy_coef = .005
    class runner(Go2HIMCfgPPO.runner):
        policy_class_name = 'BoundedHIMActorCritic'
        experiment_name = 'go2_him_retrain'
        num_steps_per_env = 32
        max_iterations = 4000
        save_interval = 50
        resume = False
