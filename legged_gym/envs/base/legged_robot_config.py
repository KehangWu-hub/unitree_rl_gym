from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None # 非None时，step()返回用于非对称训练的Critic特权观测；否则返回None
        num_actions = 12
        env_spacing = 3.  # 使用高度场或三角网格时不使用该参数
        send_timeouts = True # 向算法发送超时信息
        episode_length_s = 20 # 单回合时长，单位：秒
        test = False

    class terrain:
        mesh_type = 'plane' # 可选：none、plane、heightfield或trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # 仅用于崎岖地形：
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1米×1.6米矩形区域（不含中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # 是否只选择一种地形并传入其全部参数
        terrain_kwargs = None # 所选地形的参数字典
        max_init_terrain_level = 5 # 课程学习的初始地形等级
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # 地形行数（难度等级）
        num_cols = 20 # 地形列数（类型）
        # 地形类型：[平滑斜坡、崎岖斜坡、上楼梯、下楼梯、离散障碍]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # 仅用于三角网格：
        slope_treshold = 0.75 # 超过该阈值的斜坡会被修正为垂直表面

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # 默认：x/y线速度、偏航角速度、航向角；航向模式下会由航向误差重算偏航角速度
        resampling_time = 10. # 重新采样指令的时间间隔，单位：秒
        heading_command = True # 为True时，根据航向误差计算角速度指令
        class ranges:
            lin_vel_x = [-1.0, 1.0] # 最小值、最大值，单位：米/秒
            lin_vel_y = [-1.0, 1.0]   # 最小值、最大值，单位：米/秒
            ang_vel_yaw = [-1, 1]    # 最小值、最大值，单位：弧度/秒
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # 动作为0时的目标关节角
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P：位置，V：速度，T：力矩
        # PD驱动参数：
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # 动作缩放：目标角度 = 动作缩放系数 * 动作 + 默认角度
        action_scale = 0.5
        # 降采样倍数：每个策略周期包含的仿真步数
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # 仿真中的Actor名称
        foot_name = "None" # 足部刚体名称，用于索引刚体状态和接触力张量
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # 合并由固定关节连接的刚体；可用dont_collapse="true"保留指定固定关节
        fix_base_link = False # 固定机器人基座
        default_dof_drive_mode = 3 # 参见GymDofDriveModeFlags：0无驱动、1位置目标、2速度目标、3力驱动
        self_collisions = 0 # 位掩码过滤器：1表示禁用自碰撞，0表示启用
        replace_cylinder_with_capsule = True # 用胶囊体替换碰撞圆柱体，使仿真更快、更稳定
        flip_visual_attachments = True # 某些.obj网格必须从y轴向上翻转为z轴向上
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # 为True时将负的总奖励截断为0，以避免训练早期终止问题
        tracking_sigma = 0.25 # 跟踪奖励 = exp(-误差平方/sigma)
        soft_dof_pos_limit = 1. # URDF关节限位的比例，超过该范围会受到惩罚
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # 超过该值的接触力会受到惩罚

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # 对其他噪声数值进行整体缩放
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # 查看器相机：
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0表示y轴向上，1表示z轴向上

        class physx:
            num_threads = 10
            solver_type = 1  # 0：PGS，1：TGS
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 # 8000个及以上环境需要设为2**24
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0：不收集，1：最后一个子步，2：所有子步（默认值）

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # 可选：elu、relu、selu、crelu、lrelu、tanh、sigmoid
        # 仅用于ActorCriticRecurrent：
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # 训练参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # 小批量大小 = 环境数×每环境步数/小批量数
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # 可选adaptive或fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # 每轮迭代中每个环境采集的步数
        max_iterations = 1500 # 策略更新次数

        # 日志记录
        save_interval = 50 # 每隔该迭代数检查并保存一次
        experiment_name = 'test'
        run_name = ''
        # 加载与续训
        resume = False
        load_run = -1 # -1表示最后一次运行
        checkpoint = -1 # -1表示最后保存的模型
        resume_path = None # 根据load_run和checkpoint更新
