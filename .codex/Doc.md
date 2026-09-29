# 第一大阶段：从 Isaac Gym 训练到 MuJoCo 验证

本文用于理解并复现本项目当前的第一大阶段：

```text
Isaac Gym 训练 → Isaac Gym Play/导出 → MuJoCo Sim2Sim
```

当前有效基线是 **45维 Actor / 60维 Critic / 12维动作** 的 Go2 策略，实验名为
`rough_go2_45x60_rewards`。Actor只使用真机可获得的信息；Critic只在训练时额外使用
仿真器特权信息。MuJoCo和真机只读策略影子检查使用导出的45维Actor；真机尚未下发电机指令。

截至2026-09-29，已完成实体Go2 EDU有线只读连接和12秒实时策略影子检查，尚未验收真机电机控制。键盘速度输入限定在只读影子检查与DDS仿真中使用，App虚拟摇杆的数据来源仍待核对。

## 一、先理解整体数据流

```text
go2_config.py
  定义机器人、观测维度、控制参数、奖励权重和PPO实验名
        ↓
go2_env.py + legged_robot.py
  创建Isaac Gym环境、构造45/60维观测、计算奖励并执行动作
        ↓
train.py + rsl_rl
  PPO训练，生成model_<iteration>.pt checkpoint
        ↓
play.py
  加载checkpoint，在Isaac Gym回放并导出Actor和deploy.yaml
        ↓
deploy/common/go2_policy.py
  统一定义部署端45维观测、关节顺序和动作到目标关节角的转换
        ↓
deploy/deploy_mujoco/deploy_go2.py
  加载官方Go2 MJCF、TorchScript Actor和deploy.yaml，在MuJoCo闭环运行
```

## 二、理解 Isaac Gym 训练需要阅读的文件

建议按以下顺序阅读。

### 1. Go2任务注册

文件：`legged_gym/envs/__init__.py`

关键代码是：

```python
task_registry.register("go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())
```

它把命令行中的 `--task=go2` 连接到Go2环境类、环境配置和PPO配置。

### 2. Go2训练配置

文件：`legged_gym/envs/go2/go2_config.py`

重点理解：

- `env.num_observations = 45`：Actor输入45维。
- `env.num_privileged_obs = 60`：Critic输入60维，只用于训练。
- `init_state.default_joint_angles`：动作等于零时的默认关节角。
- `control.stiffness/damping`：PD参数，目前为Kp=20、Kd=0.5。
- `control.action_scale = 0.25`：策略动作转换成关节角增量的比例。
- `control.decimation = 4`：每4个物理步更新一次策略。
- `asset.file`：Isaac Gym训练使用的Go2 URDF。
- `rewards.scales`：Go2启用的奖励及权重。
- `runner.experiment_name = rough_go2_45x60_rewards`：训练与导出目录名。

Go2当前启用的奖励包括速度跟踪、竖直速度、横滚/俯仰角速度、力矩、关节速度、
关节加速度、动作变化、关节限位、能耗、平直姿态、默认关节姿态、腾空时间和非期望接触。

### 3. Go2专用环境

文件：`legged_gym/envs/go2/go2_env.py`

重点是 `Go2Robot.compute_observations()`：

Actor的45维观测按以下顺序拼接：

```text
机身角速度3
+ 投影重力3
+ 速度指令3
+ 12个关节位置偏差
+ 12个关节速度
+ 上一次动作12
= 45维
```

Critic的60维观测为：

```text
机身线速度3
+ 机身角速度3
+ 投影重力3
+ 速度指令3
+ 关节位置偏差12
+ 关节速度12
+ 关节力矩12
+ 上一次动作12
= 60维
```

`_get_noise_scale_vec()` 定义Actor各段观测的训练噪声。Critic观测不导出，也不能在
MuJoCo或真机运行时误接到Actor。

### 4. 通用腿式机器人环境和奖励公式

文件：`legged_gym/envs/base/legged_robot.py`

它实现仿真步进、PD控制、奖励装载、重置、指令采样和通用奖励函数。Go2与其他机器人
共享这里的方法，但是否启用以及权重大小由各自配置决定。

Go2新增的两个奖励也已经迁入该基类，并注明为Go2任务设计：

- `_reward_energy()`：`Σ(|关节速度| × |关节力矩|)`。
- `_reward_joint_pos()`：偏离默认姿态的L2惩罚；无指令且机身低速时放大5倍。

奖励加载器根据配置名自动寻找 `_reward_<配置名>()`。权重为零或不存在于当前机器人
配置中的奖励不会运行，因此其他机器人不会自动启用这两个Go2奖励。

### 5. 基础配置

文件：`legged_gym/envs/base/legged_robot_config.py`

这里定义所有机器人继承的默认配置，例如仿真步长、地形、命令范围、归一化、噪声、
域随机化、基础奖励参数和PPO默认网络参数。阅读Go2配置时，要同时检查它从这里继承了
哪些没有覆盖的值。

### 6. 训练入口和PPO实现

- `legged_gym/scripts/train.py`：创建环境与PPO runner并开始训练。
- `legged_gym/utils/task_registry.py`：根据任务名合并配置、创建环境、加载或新建runner。
- `rsl_rl/runners/on_policy_runner.py`：PPO采样、更新、保存checkpoint的主循环。
- `rsl_rl/algorithms/ppo.py`：PPO损失和参数更新。
- `rsl_rl/modules/actor_critic.py`：Actor/Critic网络结构。

第一次阅读不必先钻进PPO数学细节。应先确认观测、动作、奖励、控制周期和导出契约正确，
再研究算法实现。

## 三、理解从 checkpoint 到部署文件

### 1. checkpoint是什么

训练产生：

```text
logs/rough_go2_45x60_rewards/<时间>_<run名>/model_<轮数>.pt
```

checkpoint包含Actor/Critic、优化器和训练迭代状态，用于继续训练或由Play加载。它不是
MuJoCo最终直接使用的轻量策略文件。

当前正式训练run为：

```text
logs/rough_go2_45x60_rewards/Sep08_17-27-22_go2_45x60_official_rewards/
```

最终checkpoint为 `model_1500.pt`。

### 2. Play与导出

文件：`legged_gym/scripts/play.py`

Play会：

1. 创建Go2 Isaac Gym测试环境；
2. 加载指定checkpoint；
3. 在Isaac Gym中运行Actor；
4. 导出仅供推理的TorchScript Actor；
5. 导出与该策略配套的部署参数。

导出结果：

```text
logs/rough_go2_45x60_rewards/exported/policies/policy_1.pt
logs/rough_go2_45x60_rewards/exported/params/deploy.yaml
```

`policy_1.pt` 与 `deploy.yaml` 必须成对使用。不能给新策略配旧的缩放、默认关节角、
PD参数或关节顺序。

### 3. 部署契约导出

实现位于 `legged_gym/utils/deploy.py` 中的 `export_deploy_config()`。

`deploy.yaml`记录：

- Actor观测维度和动作维度；
- 45维观测各部分顺序及缩放；
- 策略关节名称顺序；
- 默认关节角；
- 动作缩放与裁剪；
- Kp、Kd和力矩限制；
- 速度指令范围；
- 策略控制周期。

## 四、理解 MuJoCo 部署需要阅读的文件

### 1. Go2公共策略接口

文件：`deploy/common/go2_policy.py`

这是Gym导出与MuJoCo、未来真机部署之间的公共契约，重点函数为：

- `load_deploy_config()`：检查部署配置版本和机器人类型。
- `name_to_indices()`：按关节名称生成映射，避免依赖偶然的数组顺序。
- `projected_gravity()`：由IMU四元数计算机身坐标系中的重力方向。
- `build_observation()`：严格按训练顺序生成45维Actor观测。
- `action_to_target()`：执行动作裁剪，并计算 `默认角度 + action × 0.25`。

未来真机程序也应复用这里的观测和动作规则，不能再手写一套不同常量。

### 2. Go2 MuJoCo运行器

文件：`deploy/deploy_mujoco/deploy_go2.py`

它负责：

- 加载MuJoCo XML、策略和 `deploy.yaml`；
- 根据关节名称定位MuJoCo的qpos、qvel和actuator；
- 从IMU和关节状态构造45维观测；
- 每0.02秒运行一次Actor；
- 将Actor输出转换为目标关节角；
- 以PD控制在0.002秒物理步长下驱动机器人；
- 对力矩限幅并检测跌倒、异常数值；
- 输出速度、位移、高度、倾角和力矩等JSON指标。

### 3. MuJoCo运行配置

文件：`deploy/deploy_mujoco/configs/go2.yaml`

它指定：

- `policy_path`：当前正式TorchScript Actor；
- `deploy_config_path`：配套部署契约；
- `xml_path`：默认平地场景 `scene_flat.xml`；
- `simulation_dt = 0.002`；
- 默认速度指令与运行时间；
- 用于匹配Isaac Gym训练模型的阻尼、armature、摩擦覆盖值；
- 跌倒高度和倾角阈值。

### 4. Go2 MJCF与资源

目录：`resources/robots/go2/mjcf/`

MJCF是MuJoCo使用的XML模型格式，描述机器人刚体、关节、惯量、碰撞、执行器、传感器和
场景引用。该目录来自官方 `unitree_mujoco` 的Go2资产。

重点文件：

- `go2.xml`：Go2机器人本体。
- `scene_flat.xml`：用于基础速度和稳定性验收的平地场景。
- `scene.xml`：官方带障碍和阶梯场景，用于进阶测试。
- mesh/纹理/地形资产：被XML引用的几何和视觉资源，不参与策略网络计算。

Isaac Gym训练使用URDF，MuJoCo使用MJCF；两者不是同一个动力学文件，因此Sim2Sim的意义
就是检查策略是否过度依赖Isaac Gym或URDF特性。

### 5. 自动测试

- `tests/test_go2_policy_interface.py`：验证45维观测、动作转换和名称映射。
- `tests/test_go2_training_contract.py`：验证Go2关键奖励公式。
- `tests/test_go2_mujoco.py`：验证官方MJCF尺寸、执行器、传感器和PD站立。

这些测试应该保留。它们不是一次性脚本，而是防止以后修改观测顺序、关节映射或XML时
悄悄破坏已验收流程的回归测试。

## 五、实际执行：直接在 MuJoCo 看当前正式效果

### 1. 进入项目和MuJoCo环境

```bash
cd /home/wkh/projects/unitree_rl_gym
source /home/wkh/anaconda3/etc/profile.d/conda.sh
conda activate unitree-rl
```

### 2. 打开MuJoCo窗口查看前进效果

```bash
python deploy/deploy_mujoco/deploy_go2.py --command 0.5 0.0 0.0
```

窗口中Go2应保持正常站高并向前移动。关闭窗口或到达配置时长后，终端会输出JSON指标。

指令顺序是：

```text
--command 前进速度 侧向速度 偏航角速度
```

常用示例：

```bash
# 静止站立
python deploy/deploy_mujoco/deploy_go2.py --command 0.0 0.0 0.0

# 向左侧移
python deploy/deploy_mujoco/deploy_go2.py --command 0.0 0.3 0.0

# 原地左转
python deploy/deploy_mujoco/deploy_go2.py --command 0.0 0.0 0.5

# 前进、侧移、右转组合
python deploy/deploy_mujoco/deploy_go2.py --command 0.5 0.2 -0.3
```

### 3. 无窗口快速验收

```bash
python deploy/deploy_mujoco/deploy_go2.py \
  --headless --no-realtime --duration 10 \
  --command 0.5 0.0 0.0
```

成功时进程返回码为0，并且JSON中应满足：

- `fell: false`
- `finite: true`
- `min_height` 未低于跌倒阈值
- `max_tilt_rad` 未超过倾倒阈值

### 4. 外力扰动测试

```bash
python deploy/deploy_mujoco/deploy_go2.py \
  --headless --no-realtime --duration 10 \
  --command 0.5 0.0 0.0 \
  --push-at 5 --push-duration 0.2 --push-force 50 0 0
```

## 六、实际执行：重新训练、Play和导出

只有在修改奖励、观测、控制参数或训练配置后才需要重训。仅重构函数所在文件而没有改变
公式、名称和权重，不需要重训。

### 1. 进入Isaac Gym训练环境

```bash
cd /home/wkh/projects/unitree_rl_gym
source /home/wkh/anaconda3/etc/profile.d/conda.sh
conda activate LeggedGym
```

### 2. 先做小规模契约训练

```bash
python legged_gym/scripts/train.py \
  --task=go2 --headless \
  --num_envs=64 --max_iterations=2 \
  --run_name=contract_check
```

检查网络打印应为Actor输入45、Critic输入60，并确认loss有限、无NaN/Inf和维度错误。

### 3. 正式训练

```bash
python legged_gym/scripts/train.py \
  --task=go2 --headless \
  --num_envs=4096 --max_iterations=1500 \
  --run_name=go2_45x60_official_rewards
```

结果保存在：

```text
logs/rough_go2_45x60_rewards/<时间>_go2_45x60_official_rewards/
```

### 4. 指定run和checkpoint进行Play并导出

将 `<实际run目录名>` 替换成训练生成的目录名称：

```bash
python legged_gym/scripts/play.py \
  --task=go2 --headless \
  --load_run=<实际run目录名> \
  --checkpoint=1500
```

如果要打开Isaac Gym窗口查看效果，去掉 `--headless`：

```bash
python legged_gym/scripts/play.py \
  --task=go2 \
  --load_run=<实际run目录名> \
  --checkpoint=1500
```

Play结束前会将Actor与部署契约覆盖导出到当前实验的 `exported/` 目录。导出后必须再次运行
MuJoCo无窗口验收和GUI观察，不能仅凭Isaac Gym效果判定可部署。

## 七、运行测试

```bash
cd /home/wkh/projects/unitree_rl_gym
source /home/wkh/anaconda3/etc/profile.d/conda.sh
conda activate unitree-rl
python -m unittest discover -s tests
```

当前Go2相关测试应为23项全部通过。系统默认Python可能没有安装MuJoCo，因此测试应在 `unitree-rl`
环境运行。

## 八、当前正式产物与注意事项

当前MuJoCo默认配置已经指向：

```text
策略：logs/rough_go2_45x60_rewards/exported/policies/policy_1.pt
契约：logs/rough_go2_45x60_rewards/exported/params/deploy.yaml
场景：resources/robots/go2/mjcf/scene_flat.xml
```

注意：

1. 不要混用不同训练run产生的 `policy_1.pt` 和 `deploy.yaml`。
2. 不要随意改变45维观测顺序、缩放、关节顺序或0.02秒策略周期。
3. MuJoCo跟踪速度不等于指令的精确伺服值；策略目标是近似跟踪并保持稳定。
4. 第一大阶段已经完成并验收；真机部署属于第二大阶段，必须先完成SDK/MuJoCo安全链路，
   不能直接把当前Python MuJoCo运行器当作真机程序。

# 第二大阶段：从 MuJoCo 到实体 Go2

本阶段接续第一大阶段，目标是把已经通过直接MuJoCo验证的策略接入Unitree SDK2与DDS控制链路，
先完成官方DDS/MuJoCo闭环，再按安全顺序推进实体Go2验收。

当前第二大阶段已经完成控制软件、离线测试和官方DDS/MuJoCo闭环；以下内容既是学习路线，也是后续实体机操作依据。

## 一、从直接MuJoCo到真机：先理解差别

第一阶段的直接MuJoCo运行器是：

```text
deploy/deploy_mujoco/deploy_go2.py
        ↓ 直接读取MuJoCo内存中的状态
45维观测 → Actor → 12维动作
        ↓ 直接写入MuJoCo控制量
MuJoCo物理仿真
```

它验证了策略、45维观测、关节映射、动作缩放、PD控制和MuJoCo动力学，但没有验证真机通信。

最终Real程序是：

```text
deploy/deploy_real/deploy_real_go2.py
        ↓ 发布LowCmd
Unitree SDK2 → DDS → 官方unitree_mujoco或实体Go2
        ↑ 订阅LowState
```

这条链路额外验证：

- `unitree_sdk2py`能否正确创建DDS发布器和订阅器；
- `LowState`中的IMU和关节数据能否正确读取；
- `LowCmd`中的目标角、Kp、Kd、电机模式和CRC能否正确发送；
- 策略关节顺序与SDK电机顺序是否一致；
- 20毫秒控制循环是否稳定；
- 通信或策略异常时能否自动进入阻尼。

所谓“DDS闭环验证”，就是先用官方 `unitree_mujoco` 代替实体Go2，但控制端运行真正的Real程序。
将来连接实体机器人时，只替换DDS另一端，控制端程序和消息接口不变。

## 二、需要按顺序读懂的文件

### 1. 正式部署模型

文件：

```text
deploy/pre_train/go2/motion.pt
```

这是已经通过Isaac Gym、直接MuJoCo和官方DDS闭环验证的45维TorchScript Actor。它执行：

```text
输入：(1, 45)
输出：(1, 12)
```

`pre_train/go2/` 中只保存这个PT模型，不保存YAML。训练checkpoint仍位于 `logs/`，但Real运行只加载
冻结后的 `motion.pt`。

### 2. MuJoCo和Real公共策略接口

文件：

```text
deploy/common/go2_policy.py
```

重点阅读：

- `projected_gravity()`：把世界重力方向投影到机身坐标系；
- `build_observation()`：严格按训练顺序拼接45维Actor观测；
- `action_to_target()`：把12维动作转换成目标关节角；
- `name_to_indices()`：根据关节名称生成顺序映射；
- `resolve_path()`：展开配置中的项目根目录占位符。

该文件保留在 `deploy/common/`，是因为直接MuJoCo运行器和Real程序确实共同调用它。修改这里会同时
影响MuJoCo与真机，因此修改后必须重新执行两边的测试。

### 3. Go2 Real配置

文件：

```text
deploy/deploy_real/configs/go2.yaml
```

它只描述Real部署，不替代 `deploy/deploy_mujoco/configs/go2.yaml`。两个同名文件位于不同运行环境，
内容和职责不同。

Real配置需要理解的分区：

- `policy_path`：冻结的 `motion.pt` 路径；
- `step_dt`：策略周期0.02秒，即50 Hz；
- `policy`：45维输入、12维输出、策略关节顺序、SDK顺序和映射；
- `observations`：45维各部分的缩放；
- `actions`：默认关节角、动作缩放和裁剪；
- `control`：Policy阶段Kp/Kd与力矩限制；
- `dds`：domain、LowCmd/LowState topic和订阅队列；
- `command_limits`：键盘速度输入使用的保守速度上限；
- `fsm`：FixStand时间、站立增益、阻尼增益；
- `safety`：通信、姿态、关节、动作与力矩阈值；
- `recording`：JSONL运行记录设置。

### 4. Go2 Real主程序

文件：

```text
deploy/deploy_real/deploy_real_go2.py
```

这是需要重点读懂的文件。建议不要从头连续阅读，而是按以下类和函数顺序阅读。

#### `Go2RealConfig`

负责读取Real `go2.yaml`，校验：

- 机器人必须是Go2；
- 配置格式版本正确；
- Actor必须为45维输入、12维输出；
- `policy_to_sdk`必须是0到11的完整排列；
- 控制周期、默认角、Kp/Kd和力矩限制必须有效。

#### `Go2FSM`

管理四个控制状态：

```text
Passive → FixStand → Policy
    \          \         \
     └──────────┴─────────→ Damping
```

- `Passive`：程序初始状态，不允许直接执行策略；
- `FixStand`：用三次平滑插值在5秒内移动到默认站姿；
- `Policy`：以50 Hz执行强化学习策略；
- `Damping`：用户急停或任何异常后的安全降级状态。

FixStand使用：

```text
Kp = 40
Kd = 1
```

Policy仍使用训练配置：

```text
Kp = 20
Kd = 0.5
```

两套增益不能混淆。前者负责从初始姿态可靠站起，后者必须与策略训练时的控制条件一致。

#### `Go2Safety`

在发送任何策略目标前检查：

- LowState是否超时；
- 控制循环是否超时；
- 数值是否包含NaN或Inf；
- 横滚和俯仰是否超过限制；
- 关节位置和速度是否越界；
- 策略动作绝对值是否过大；
- 当前动作相对上一帧是否突变；
- 目标关节角是否越界；
- 估算PD力矩是否超过限制。

估算力矩为：

```text
tau_est = Kp × (q_target - q) - Kd × dq
```

任一检查失败后，策略目标不会继续下发，FSM会切换到Damping。

#### `Go2State`与`Go2Command`

`Go2State`保存SDK顺序的单帧状态：

```text
时间戳、IMU四元数、机身角速度、12关节角、12关节速度
```

`Go2Command`保存SDK顺序的待发送命令：

```text
状态机模式、12目标关节角、12个Kp、12个Kd、故障原因
```

把状态和命令定义成明确的数据结构，可以避免控制逻辑直接依赖DDS消息对象。

#### `Go2Controller`

这是策略控制核心，但它自身不创建DDS发布器。每个周期执行：

```text
SDK顺序关节状态
→ 转换为策略顺序
→ 检查状态
→ 构造45维观测
→ TorchScript推理
→ 动作转换为目标关节角
→ 检查动作、目标角和估算力矩
→ 转回SDK顺序
→ 返回Go2Command
```

这种设计使Controller可以用假策略和数组完成离线单元测试，不必连接机器人。

#### `Go2DDSTransport`

负责真正的Unitree通信：

- 创建 `rt/lowstate` 订阅器；
- 把LowState转换为 `Go2State`；
- 创建 `rt/lowcmd` 发布器；
- 把 `Go2Command`写入12个电机命令；
- 计算并写入CRC；
- 发送LowCmd；
- 启动发布前探测是否已有其他LowCmd写入者。

只读模式不会创建LowCmd发布器，因此可以先检查实体机器人状态而不发送电机命令。

#### `Go2Recorder`

把每个周期的数据保存到：

```text
logs/go2_real/go2_real_<时间>.jsonl
```

记录内容包括IMU、关节状态、速度指令、目标角、Kp/Kd、FSM状态和安全故障原因。真机出现异常时，
应先检查记录，而不是直接放宽安全阈值。

#### `run()`与`parse_args()`

`parse_args()`定义命令行参数；`run()`负责把配置、Controller、DDS Transport和Recorder组织成完整程序。
主动DDS仿真模式的`finally`路径会发送约1秒阻尼命令；只读模式不发布电机命令。

### 5. 键盘速度输入

`deploy_real_go2.py`中的`KeyboardVelocity`把`W/S`、`A/D`、`Q/E`映射为前后、侧向、偏航速度。`X`或空格把速度目标归零，策略继续运行；0.25秒未收到方向按键时也会归零。`Z`或`Ctrl+C`退出，仿真程序退出时发送约1秒阻尼命令，机器人可能降低机身。键盘仅用于DDS仿真和真机只读影子检查，真机主动控制仍被启动保护阻止。

### 6. Unitree Python SDK

SDK源码位于当前项目同级目录：

```text
../unitree_sdk2_python/unitree_sdk2py/
```

建议重点了解：

```text
core/channel.py                 DDS初始化、Publisher和Subscriber
idl/unitree_go/msg/dds_.py      Go2 LowState/LowCmd消息类型
utils/crc.py                    LowCmd CRC计算
```

不需要从头阅读整个SDK，只需理解本项目实际调用的接口：

```python
ChannelFactoryInitialize(domain_id, network)
ChannelSubscriber(topic, message_type)
ChannelPublisher(topic, message_type)
publisher.Write(message)
CRC().Crc(low_cmd)
```

### 7. 官方unitree_mujoco DDS桥接

官方仓库中的关键文件：

```text
unitree_mujoco/simulate_python/unitree_mujoco.py
unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py
unitree_mujoco/simulate_python/config.py
```

`unitree_mujoco.py`运行物理仿真；`unitree_sdk2py_bridge.py`负责：

- 将MuJoCo传感器状态封装成LowState并发布；
- 订阅LowCmd；
- 将目标角、Kp、Kd和前馈力矩转换成MuJoCo执行器力矩。

因此DDS仿真不是直接调用本项目的MuJoCo运行器，而是两个独立程序通过DDS交换消息。

## 三、必须理解的DDS概念

### 1. Publisher、Subscriber和Topic

DDS采用发布—订阅通信：

```text
官方MuJoCo或实体Go2 --发布rt/lowstate--> Go2 Real程序
Go2 Real程序          --发布rt/lowcmd----> 官方MuJoCo或实体Go2
```

`rt/lowstate`包含IMU和关节等状态；本程序使用这些状态字段。`rt/lowcmd`包含电机目标角、速度、Kp、Kd、前馈力矩、
模式和CRC。

### 2. Domain ID

只有相同DDS domain中的程序才能互相发现。当前约定：

```text
官方MuJoCo仿真：domain 1
实体Go2：通常为domain 0
```

仿真使用非0 domain，是为了避免仿真LowCmd误发给同一网络中的实体机器人。

### 3. 网卡

仿真使用：

```text
lo
```

即本机回环网卡，消息不会离开当前电脑。实体Go2使用与机器人网线连接的实际网卡，例如
`enp3s0`。

### 4. CRC

CRC是消息完整性校验，不是DDS本身。每次发送LowCmd前都要重新计算CRC。错误或过期的CRC可能导致
机器人拒绝命令。

## 四、必须理解的关节映射

Actor训练使用的策略顺序是：

```text
FL → FR → RL → RR
每条腿：hip → thigh → calf
```

Unitree SDK电机顺序是：

```text
FR → FL → RR → RL
每条腿：hip → thigh → calf
```

因此策略读取SDK状态时使用：

```text
policy_to_sdk = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
```

状态方向：

```text
q_policy = q_sdk[policy_to_sdk]
```

命令方向：

```text
q_target_sdk[policy_to_sdk] = q_target_policy
```

如果只复用同一组索引却把读写方向写反，机器人会出现左右腿或前后腿错位。因此关节映射测试是
真机部署前的强制检查。

## 五、实际操作顺序

### 1. 进入正确环境

```bash
cd /home/wkh/projects/unitree_rl_gym
source /home/wkh/anaconda3/etc/profile.d/conda.sh
conda activate unitree-rl
```

`unitree_sdk2py`安装在 `unitree-rl` 环境，不能使用没有安装SDK的Base Python。

### 2. 不启动DDS，只检查Real配置与模型

```bash
python deploy/deploy_real/deploy_real_go2.py --check
```

该命令验证：

- Real `go2.yaml`可以加载；
- `motion.pt`可以加载；
- 模型接受45维输入；
- 模型返回12维有限动作；
- 全程不创建DDS发布器。

### 3. 运行全部Go2测试

```bash
python -m unittest discover -s tests -p 'test_go2*.py'
```

当前应为23项全部通过。

### 4. 官方DDS/MuJoCo闭环

先在官方 `unitree_mujoco/simulate_python/config.py` 中确认：

```text
ROBOT = "go2"
DOMAIN_ID = 1
INTERFACE = "lo"
```

第一终端启动官方仿真器：

```bash
cd <unitree_mujoco目录>/simulate_python
python unitree_mujoco.py
```

第二终端启动真正的Real程序：

```bash
cd /home/wkh/projects/unitree_rl_gym
conda activate unitree-rl
python deploy/deploy_real/deploy_real_go2.py lo \
  --domain-id 1 \
  --simulation-auto \
  --duration 12 \
  --command 0.3 0 0
```

`--simulation-auto`被程序限制为 `lo + 非0 domain`，不能用于实体机器人。该命令自动执行5秒FixStand，
再执行7秒Policy。

当前前进DDS回归结果：601帧、平均周期20.000毫秒、最大周期20.055毫秒、零安全故障。

组合指令回归：

```bash
python deploy/deploy_real/deploy_real_go2.py lo \
  --domain-id 1 \
  --simulation-auto \
  --duration 12 \
  --command 0.2 0.1 -0.2
```

当前组合回归结果：601帧、平均周期20.000毫秒、最大周期22.692毫秒、零安全故障。

### 5. 实体Go2只读连接

2026-09-28，已将Go2 EDU背部标注 `RJ45` 的接口与笔记本有线网卡 `eno1` 用网线连接。
电脑原有的 `DSL 连接 1` 是PPPoE拨号配置；另建了独立的NetworkManager连接 `go2-direct`，
将 `eno1` 设为 `192.168.123.99/24`，不设置默认路由，未改动原有拨号配置。
网口物理连接正常，`eno1` 已启用该地址。

本地检查 `python deploy/deploy_real/deploy_real_go2.py --check` 成功；后续新增只读影子检查与真机控制防误启动测试后，Go2自动测试20项全部通过。
随后在 `unitree-rl` 环境运行：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 --read-only
```

程序成功输出“已连接LowState”和“当前为只读模式，不创建LowCmd发布器”。
约11.1秒内记录556条状态，均为 `passive`，每条都有12个关节状态，安全故障原因均为空。
本次记录：`logs/go2_real/go2_real_20260928_220913.jsonl`。本次没有向实体机器人发送电机命令。

这次仅确认DDS收到LowState和12关节数据；尚未独立核对IMU方向、关节符号、上层控制输入来源与状态更新频率。
只读模式不创建LowCmd发布器。进入任何实体电机控制测试前，仍需验证控制输入、独立停机方式、无其他LowCmd发布者，
并按官方流程处理高层运动服务 `sport_mode`。悬空支撑可降低首次测试风险；若无支撑设备，地面测试需要另行设计受限控制并考虑跌倒风险。不能直接跳到Policy。

2026-09-29，真机网卡上不带`--read-only`的运行被启动前保护阻止，避免停机方式未验收时误发LowCmd；手机App可用于原厂运动控制，但尚未接入本策略程序。加入`--shadow-policy`后，在已连接的`eno1`上运行：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 --read-only --shadow-policy --duration 12
```

程序输出“不创建LowCmd发布器”，完成5秒虚拟FixStand并进入`Policy=True`，12秒后自动退出。日志为`logs/go2_real/go2_real_20260929_141828.jsonl`：601帧，其中`fix_stand`250帧、`policy`351帧；周期中位数20.0毫秒、最大20.868毫秒；状态年龄最大7.578毫秒；安全故障为空，记录数值均有限，速度指令全为零。策略目标与实际关节角最大差约0.4205弧度。这证明实时真机观测可完成策略计算和软件安全检查；没有验证任何真机电机控制、步态或急停。当天另以9月28日的556帧日志离线回放，得到306帧策略输出且无软件安全故障。

### 6. 实体机后续分级顺序

必须依次完成：

```text
只读LowState
→ 策略只推理、不下发
→ 键盘输入与独立停机方案的仿真验收
→ 受限电机控制与急停验收
→ FixStand验收
→ 落地站立
→ 微速前进与停止
→ 后退、侧移、转向
→ 组合指令
→ 轻扰动与长时运行
```

任一步出现异常，都应停止升级测试并检查JSONL记录，不能通过放宽安全阈值掩盖问题。

## 六、当前完成边界

当前已经完成：

- 同一45维Actor从Isaac Gym导出到直接MuJoCo；
- Go2 Real程序、Real配置、四态FSM和安全控制器；
- Unitree SDK2、DDS、LowState、LowCmd与CRC接入；
- 官方 `unitree_mujoco` 前进和组合指令闭环；
- 23项Go2自动测试；
- Go2专用LowCmd电机模式已按宇树Go2低层示例改为`0x01`，离线契约测试覆盖初始化和发送；
- 正式模型冻结到 `deploy/pre_train/go2/motion.pt`；
- 实体Go2 EDU有线网络配置及只读LowState连接，未下发电机命令。
- 真机状态下12秒只读策略影子检查，进入Policy并记录351帧策略输出，未下发电机命令。
- 键盘前进与`X`/空格归零的动态DDS/MuJoCo验收；修复主循环复制状态前取时间导致的误报超时。

当前尚未完成：

- 实体IMU、关节方向、App虚拟摇杆数据来源与数据更新频率的逐项核对；
- 独立停机路径的仿真和真机验证；
- 实体阻尼、受限电机控制与FixStand验收；
- 实体策略下发；
- 落地运动与长时验收。

因此当前准确表述应为：

> 已完成Go2控制软件、官方DDS/MuJoCo闭环、实体Go2有线只读连接和实时策略影子检查；尚未向实体机器人发送电机命令，不能宣称已经完成真机运动部署。

### 2026-09-29 上层指令接口决策

本项目的45维Actor观测中包含三维速度指令，即前进、横移和偏航角速度。键盘路径调用`Go2Controller.set_velocity_command()`。手机App能控制宇树原厂运动功能，不代表它的虚拟摇杆已接入当前程序；实体低层控制还需要按官方流程处理原厂运动服务。

`--keyboard`仅能与`--simulation-auto`或`--read-only --shadow-policy`配合使用；`W/S`、`A/D`、`Q/E`分别给出前后、侧向、转向速度，`X`与空格将速度目标归零并保持策略运行。`Z`或`Ctrl+C`退出并在仿真中进入阻尼。若0.25秒未收到有效方向按键，速度目标归零。终端不提供可靠的松键事件，连续按住依赖系统按键重复。App方案先做只读数据核对，再判断是否能提供同样的速度目标。零速度是停止行走指令，不能代替已验证的独立停机措施。当前真机网卡上的主动控制启动保护继续保留，直至对应输入和停机验收完成。

静态Go2状态桥接曾验证键盘到DDS控制循环、策略观测与日志的接线，记录为`logs/go2_real/go2_real_20260929_153556.jsonl`。随后完成动态MuJoCo测试；测试结果见下节。该静态测试属于改键前的历史记录，不再作为空格操作说明。

### 2026-09-29 动态DDS/MuJoCo键盘验收

首次动态测试在`FixStand`初期误报`LowState通信超时`。原因是主循环先取时间，再复制由DDS回调更新的状态；在两步之间更新的状态时间戳可能略晚于主循环时间。将时间采样移到状态复制之后，消除了该竞态，未修改安全阈值。

修正后使用宇树官方Python DDS桥接、Go2 MJCF、与训练匹配的MuJoCo关节参数以及`lo`网卡、domain 1运行动态仿真。键盘两段`W`前进，分别用`X`和空格归零，最后用`Z`退出。`logs/go2_real/go2_real_20260929_173459.jsonl`记录609帧：`fix_stand`250帧、`policy`359帧，软件安全故障为零。机身前进约0.95米；归零并保持策略运行期间，约10.0至11.5秒的机身高度维持在0.32米，前后位置约0.946至0.940米。`Z`触发阻尼退出后，仿真机身降低；这不是保持站立的停车方式。

以上完成键盘前进、归零停走和程序退出的动态仿真验收。真机电机控制与独立停机仍未验收，不能据此解除真机只读限制。
