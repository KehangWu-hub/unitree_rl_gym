from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 45
        # Actor只接收真机可获取的信号；Critic在训练时保留仅仿真器可获取的机身线速度。
        num_privileged_obs = 60

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # 动作为0时的目标关节角，单位：弧度
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD驱动参数：
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # 动作缩放：目标角度 = 动作缩放系数 * 动作 + 默认角度
        action_scale = 0.25
        # 降采样倍数：每个策略周期包含的仿真步数
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 位掩码过滤器：1表示禁用自碰撞，0表示启用
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_vel = -0.001
            dof_acc = -2.5e-7
            torques = -0.0002
            action_rate = -0.1
            dof_pos_limits = -10.0
            energy = -2e-5
            orientation = -2.5
            joint_pos = -0.7
            feet_air_time = 0.1
            collision = -1.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_45x60_rewards'

  
