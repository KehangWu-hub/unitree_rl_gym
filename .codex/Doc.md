# 第一大阶段：从 Isaac Gym 训练到 MuJoCo 验证

本文用于理解并复现本项目当前的第一大阶段：

```text
Isaac Gym 训练 → Isaac Gym Play/导出 → MuJoCo Sim2Sim
```

当前有效基线是 **45维 Actor / 60维 Critic / 12维动作** 的 Go2 策略，实验名为
`rough_go2_45x60_rewards`。Actor只使用真机可获得的信息；Critic只在训练时额外使用
仿真器特权信息。MuJoCo和真机只读策略影子检查使用导出的45维Actor；真机尚未下发电机指令。

截至2026-09-30，已完成实体Go2 EDU有线只读连接和12秒实时策略影子检查，以及独立软件看门狗和受限单关节测试的六种动态DDS/MuJoCo仿真验证。尚未验收真机电机控制和停机。键盘速度输入限定在只读影子检查与DDS仿真中使用，App虚拟摇杆的数据来源仍待核对。

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

当前Go2相关离线测试为27项通过，DDS集成测试默认跳过。系统默认Python可能没有安装MuJoCo，因此测试应在 `unitree-rl`
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
主动DDS仿真由独立看门狗进程创建该发布器；主进程的Transport仅订阅状态，将计算出的命令通过本机接口交给看门狗。

#### `Go2Recorder`

把每个周期的数据保存到：

```text
logs/go2_real/go2_real_<时间>.jsonl
```

记录内容包括IMU、关节状态、速度指令、目标角、Kp/Kd、FSM状态和安全故障原因。真机出现异常时，
应先检查记录，而不是直接放宽安全阈值。

#### `run()`与`parse_args()`

`parse_args()`定义命令行参数；`run()`负责把配置、Controller、DDS Transport和Recorder组织成完整程序。
主动DDS仿真模式的`finally`路径会通知独立看门狗停止，由看门狗发送约1秒阻尼命令；主进程卡住或被强制结束时，看门狗由100毫秒命令超时触发同一停止路径。只读模式不发布电机命令。

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

当前为27项离线测试通过，DDS集成测试默认跳过。运行六种动态DDS/MuJoCo故障场景：

```bash
GO2_DDS_TESTS=1 python -m unittest discover -s tests -p test_go2_watchdog.py -v
```

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
→ 键盘输入、受限电机测试与独立停机的仿真验收（已通过）
→ 真机受限电机控制与停机验收（待完成）
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
- 27项Go2离线测试及包含六种场景的DDS集成测试；
- Go2专用LowCmd电机模式已按宇树Go2低层示例改为`0x01`，离线契约测试覆盖初始化和发送；
- 正式模型冻结到 `deploy/pre_train/go2/motion.pt`；
- 实体Go2 EDU有线网络配置及只读LowState连接，未下发电机命令。
- 真机状态下12秒只读策略影子检查，进入Policy并记录351帧策略输出，未下发电机命令。
- 键盘前进与`X`/空格归零的动态DDS/MuJoCo验收；修复主循环复制状态前取时间导致的误报超时。
- 独立软件看门狗、第二终端停止和受限单关节测试的动态DDS/MuJoCo验证。

当前尚未完成：

- 实体IMU、关节方向、App虚拟摇杆数据来源与数据更新频率的逐项核对；
- 独立停机路径的真机验证；
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

### 2026-09-30 受限电机测试与独立停机

新增`Go2MotorTest`，通过`--motor-test`执行不加载策略的单关节小幅往返。默认偏移0.01弧度、时长2秒；最大偏移0.02弧度，时长1至3秒，Kp=5、Kd=0.5；实际位移、速度和估算PD力矩超限时锁存阻尼。该入口仍仅允许`lo`和非零domain，实体网卡上主动下发继续被禁止。

新增`deploy/deploy_real/go2_watchdog.py`。主动DDS仿真的主程序仅订阅LowState并计算目标，由独立看门狗进程唯一发布LowCmd。命令100毫秒未更新、LowState超时、安全检查失败、运行到期或第二终端请求停止时锁存阻尼；后续命令不能恢复主动控制。另一终端使用`python -m deploy.deploy_real.go2_watchdog --stop --domain-id 1`，收到看门狗确认后才报告成功。

2026-09-30，在`unitree-rl`环境、MuJoCo 3.2.3下完成验证。全部DDS参与者仅使用本机`lo`网卡和非零domain，没有连接实体机器人。本次不依赖外部`unitree_mujoco`工作目录；`tests/go2_dds_fixture.py`使用项目已有Go2 MJCF和官方SDK消息提供无窗口动态测试桥接。

运行命令：

```bash
GO2_DDS_TESTS=1 python -m unittest discover -s tests -p test_go2_watchdog.py -v
```

六种动态场景均通过。延迟从触发操作开始，量到测试桥接实际收到首帧Kp=0、Kd=3的LowCmd。

| 场景 | 首帧阻尼延迟 | 阻尼持续时间 | 结果 |
| --- | ---: | ---: | --- |
| 单关节往返 | 到期自动停止 | 1.019秒 | FL_hip目标峰值0.010000弧度；其余目标不变，实际关节有响应 |
| 地面低增益测试 | 偏移超限自动停止 | 1.020秒 | 实际位移超过0.04弧度后进入阻尼 |
| 第二终端停止 | 43.54毫秒 | 1.020秒 | 停止锁存，主程序收到反馈并退出 |
| 主程序SIGSTOP冻结 | 118.81毫秒 | 1.000秒 | 独立进程因心跳超时发送阻尼；恢复主程序后仍不能继续控制 |
| 主程序SIGKILL结束 | 101.48毫秒 | 1.020秒 | 主程序没有执行退出代码，独立进程仍发送阻尼 |
| LowState断流 | 118.65毫秒 | 1.020秒 | 独立状态检查触发停止 |

所有收到的命令CRC正确，12个电机模式均为`0x01`；每个场景进入阻尼后均未再次收到正Kp命令。默认离线测试另有27项通过；DDS集成测试默认跳过，需使用上面的环境变量启用。

方向测试固定机身并关闭重力，只验证映射与关节响应。地面测试保留重力，低增益控制使关节偏移超限后进入阻尼，不能据此认为无支撑真机试验安全。策略停机测试使用正式冻结模型、原有5秒FixStand和零速度指令，没有放宽现有安全阈值。

这些时间是本次本机测量，不能作为真机或其他负载条件下的最坏延迟保证。软件看门狗不能覆盖电脑断电、其自身被强制结束或通信完全中断；实体停机、退出后的姿态与电机控制仍待验收，真机网卡上的主动控制仍被程序阻止。下一步是核对实体状态、原厂运动服务切换、实际停机条件与受限测试摆放方案。

以上完成键盘前进、归零停走、受限单关节测试和独立软件停机的动态仿真验收。真机电机控制与独立停机仍未验收，不能据此解除真机只读限制。

### 2026-09-30 Go2 专用化清理

删除G1、H1、H1_2的训练环境、URDF/MJCF与网格资产、预训练策略、MuJoCo/Real配置，以及G1 C++部署目录。旧`deploy_real.py`、`config.py`、`deploy_real/common/`和旧人形机器人部署说明仅供这三种机器人使用，一并删除；旧`deploy_mujoco.py`使用人形机器人观测和配置格式，也不属于当前Go2流程，已删除。共删除244个受版本控制的文件，清除它们的任务注册和导入，保留Go2专用入口、公共训练框架、`deploy/common/go2_policy.py`和许可证。

清理后的验证：27项Go2离线测试通过；`deploy_real_go2.py --check`通过；直接MuJoCo零速度运行3秒，数值有限、未跌倒；Isaac Gym在8个Go2环境中完成一轮训练，Actor为45维、Critic为60维。小规模训练使用独立的`go2_cleanup_smoke`实验目录，不覆盖正式策略。此次改动不改变真机只读限制。

本地`LeggedGym`环境同时安装了另一个`legged_gym`，直接运行脚本可能导入该版本。在项目根目录执行`export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`后，再启动训练和Play，确保加载本仓库；README已补充该步骤。

### 2026-10-09 实体只读复核与趴卧验收准备

实体Go2通过`eno1`连接，电脑地址192.168.123.99/24。原厂模式查询返回`normal`；只读诊断约500Hz接收LowState，抽查10帧CRC正确，电机丢失计数均为0。只读探针同时观察到其他LowCmd，尚未退出原厂模式，不能同时启动另一套电机控制。未操作App摇杆时wireless_remote全零，不能由此断言App摇杆不可读取。

趴卧时四个小腿实测约-2.80至-2.82弧度，低于策略状态下限-2.7427，因此安全检查拒绝进入策略。随后由App原地站立，12秒影子检查记录601帧，250帧虚拟FixStand、351帧Policy，均有限且无故障。记录`logs/go2_real/go2_real_20261009_134232_1791524552195504936.jsonl`，最大状态年龄6.85毫秒。

16秒自动终端键盘影子检查记录801帧，250帧虚拟FixStand、551帧Policy，无故障，最大状态年龄4.35毫秒。六方向输入均进入速度指令，X与空格在下一控制帧归零；停止输入约0.26秒后归零。记录`logs/go2_real/go2_real_20261009_135501_1791525301826031869.jsonl`。两次检查均不创建LowCmd发布器。之后由使用者关机。

新增`deploy/deploy_real/go2_commission.py`作为独立的实体趴卧验收入口，不加载策略、不站起。只允许阻尼或单个髋关节测试；默认偏移0.005弧度，最大0.01弧度，时长1至3秒；所选髋Kp=5，其余11关节Kp=0，测试Kd=0.5。输入需显式确认实体趴卧与允许释放原厂模式；CRC错误帧不刷新状态时间戳。连续一秒姿态与静止检查通过后查询、释放并确认原厂模式为空，再检测其他LowCmd发布者。初始机身是否接触地面仍需现场确认，姿态数据不能证明。

趴卧检查使用实测姿态匹配范围（髋±0.5、大腿1.1至1.6、小腿-2.9至-2.6弧度），仅用于这个不承重、不控制小腿位置的验收入口，不作为机械关节范围，未改变策略配置限位。主动髋关节目标使用原有限位。实际位移超过0.04弧度、速度超过2弧度/秒、估算PD力矩超过1Nm或状态/心跳超时，锁存Kp=0、Kd=3并持续约一秒。

独立验收进程负责发布指令；主终端仅提供100毫秒心跳与键盘停止。第二终端复用`go2_watchdog --stop --domain-id 0`。确认只表示锁存停止且阻尼已提交DDS，不是电机实际停止确认；无法覆盖电脑断电、验收进程自身冻结/被强制结束或通信断开。原厂模式不自动恢复，退出后需显式`--restore-normal --confirm-prone`，先检查趴卧数据与停止接口不存在，再请求并确认normal；App实际响应仍待现场检查。

离线配置检查通过，35项软件测试通过，另2项DDS集成测试默认跳过。新增离线测试覆盖主动关节唯一性、姿态/位移/速度/非有限值故障、CRC过滤、原厂模式退出失败、真实Unix停止接口及心跳消失后一秒阻尼。离线测试中的实体DDS与RPC均替换为测试对象，不是实体停机验收。

启用`GO2_DDS_TESTS=1`后，新增入口的本机DDS测试通过：lo与非零domain模拟趴卧LowState，真正使用SDK与CRC收发，原厂RPC替换为测试对象；验证只有SDK索引3有正Kp，停止后约一秒阻尼、CRC正确且未恢复正Kp。原有六种MuJoCo/DDS场景也全部通过；本次第二终端停止47.59毫秒、主程序冻结115.19毫秒、主程序强制结束119.18毫秒、状态断流109.17毫秒，均为本机测量，不是真机延迟保证。操作顺序及命令已并入`deploy/deploy_real/README_GO2.zh.md`；实体策略入口继续禁止主动下发。实体阻尼、停止、小幅动作与原厂恢复仍待下一次开机配合。

### 2026-10-09 实体趴卧阻尼与独立停止首次检查

使用者重新开机并确认趴卧，已准备自己的第二终端停止命令。只读复核原厂模式为normal、LowState约500Hz、抽查10帧CRC正确，电机lost均为0。随后首次使用`go2_commission eno1 --damping-test --confirm-prone --release-mode --duration 2`：确认退出normal，全部Kp=0，100帧测试Kd=0.5，结束52帧Kd=3；实际最大关节变化0.000303弧度，最大状态年龄3.20毫秒。记录`logs/go2_real/commission/go2_real_20261009_170109_1791536469654074928.jsonl`。使用者确认全程趴卧，无明显动作或异常声响。

第二轮仅阻尼测试由独立进程执行`go2_watchdog --stop --domain-id 0`，在到期前停止；51帧测试、51帧锁存阻尼，全部Kp=0，最大关节变化0.000237弧度。独立CLI启动至首条阻尼计算/提交记录33.46毫秒，阻尼持续1.001秒；此时间不是实体电机响应或最坏延迟。记录`logs/go2_real/commission/go2_real_20261009_170253_1791536573083854105.jsonl`。

之后执行`--restore-normal --confirm-prone`，已查询确认normal恢复。根据使用者后续补充，恢复normal本身会自动站起，随后使用者通过App卧倒，确认App动作正常、机身再次趴卧；不能将原厂站起计为自定义控制器站起验收。上述阻尼检查只验证了趴卧状态的零Kp阻尼、软件停止请求与原厂恢复；没有验证电脑断电、网线断开、验收进程冻结等情况。

### 2026-10-09 实体左前髋小幅测试：响应与方向尚未确认

趴卧、阻尼、独立停止和App原厂恢复核对完成后，执行`go2_commission eno1 --motor-test --confirm-prone --release-mode --test-joint FL_hip_joint --test-amplitude 0.005 --duration 2`。记录`logs/go2_real/commission/go2_real_20261009_171342_1791537222325314992.jsonl`：100帧MotorTest、51帧到期阻尼，无安全故障，正Kp仅SDK索引3（FL_hip），Kp=5；其他11关节Kp=0。目标偏移峰值0.005弧度，最大状态年龄3.11毫秒。

实测FL_hip相对起点最大变化仅0.000182弧度（约0.0104度），各关节最大变化不超过0.000273弧度，接近此前静止波动。使用者确认没有明显动作或异常声响。因此只能记录指令生成/提交与运行到期停止正常，不能宣称已确认电机接受指令、实际跟踪、关节映射方向或受限动作验收通过。未增加幅度或增益，未启动FixStand或实体策略。

随后再次执行`--restore-normal --confirm-prone`，查询确认normal恢复。验收进程已结束，原厂模式已恢复；恢复动作会触发原厂站起，是使用者随后用App卧倒，不是恢复命令保证趴卧。下一步先排查微小目标的实际力矩、接触负载与低层指令接收证据，再决定受限测试参数与形式；不能直接升级到自定义站起或策略下发。

### 2026-10-09 力矩反馈与发布频率对照

核对官方Go2 Python低层示例：head、level_flag、motor mode=0x01、CRC与当前格式一致；官方示例使用2毫秒发布周期，之前的验收循环使用20毫秒。上次0.005弧度/Kp=5的弹簧项峰值仅约0.025Nm，日志未包含tau_est，不能据此认定电机接收或摩擦原因。

在`VerifiedTransport`补充同帧tau_est、motor mode/lost/温度、state tick，并独立订阅LowCmd回显与CRC；`Go2Recorder`可选写入feedback。回显仅表示本机DDS收到消息，不是电机执行确认。软件与本机DDS检查通过后，在使用者确认趴卧且停止终端就绪的情况下，保留0.005弧度/Kp=5/2秒做两轮对照：

| 条件 | 实际发布频率 | FL_hip实测最大变化 | 结果 |
| --- | --- | --- | --- |
| 原20毫秒循环 | 约50Hz | 0.000151弧度 | 100帧测试、52帧到期阻尼，99条回显CRC均正确，无安全故障 |
| 目标2毫秒循环 | 均值339.6Hz，中位数402.4Hz；最大间隔6.90毫秒 | 0.000272弧度 | 679帧测试、401帧阻尼，678条回显CRC均正确，无安全故障 |

日志分别为`logs/go2_real/commission/go2_real_20261009_182535_1791541535539904883.jsonl`和`go2_real_20261009_182828_1791541708911007857.jsonl`。50Hz轮使用者确认无明显动作或异常声响，并恢复原厂模式；500Hz目标轮尚未再次恢复。两轮tau_est变化仍不足以明确证明跟踪或方向正确；较高频率也没有产生可辨认的关节动作，未增加幅度或增益。Python循环并未实际达到稳定500Hz，不能以参数值声称频率达标。

使用者明确指出恢复normal会自动站起，每次由其再用App卧倒。已修正恢复流程、历史描述和指南：恢复命令需额外`--allow-factory-stand`，操作前明确告知原厂可能站起，并先协调App卧倒；恢复前的趴卧检查不能约束恢复后的原厂动作。本程序不自动执行卧倒。当前最后一轮验收进程已退出，原厂模式尚未恢复，待使用者选择协调恢复并App卧倒或保持趴卧直接关机；不继续施加正Kp或启动策略。

随后使用者反馈App无法进入Go模式、卧倒页面打不开，确认机身仍趴卧。只读诊断`logs/go2_real/go2_readonly_diagnostic_20261009_183345.json`查询到模式名称为空、观测LowCmd计数0，状态约500Hz且抽查CRC正确；与原厂模式尚未恢复一致，不能宣称App本身损坏或机器人失联。在明确告知恢复可能自动站起后，执行`--restore-normal --confirm-prone --allow-factory-stand`，已查询确认normal恢复。使用者随后确认Go页面已恢复、机器狗已用App卧倒，完成本轮收尾。新增自动站起确认保护后，36项软件测试通过，2项DDS集成默认跳过；新增入口本机DDS检查另已通过。当前原厂模式正常，测试进程结束，关节响应/方向与实体策略仍待验收。

### 2026-10-09 响应诊断方案与判据准备（未执行真机）

在使用者关机后，电脑端新增`--response-test`与`--analyze JSONL`，均并入已有`go2_commission.py`，没有增加文档目录。方案固定3秒、±0.005弧度平滑往返，仅所选髋Kp=20，其余Kp=0，所有Kd=0.5。五个阶段为基线0.4s、正向0.8s、基线0.4s、负向0.8s、基线0.6s。候选目标弹簧项峰值约0.1Nm，比此前0.025Nm更易辨认；不能以此保证实际动作或安全。尚未用于实体机器人，下一次执行前需现场协调这一增益变化。

本方案实际位移限值收紧为0.01弧度，速度0.5弧度/秒，PD估算力矩0.2Nm，所选关节反馈估算力矩绝对值0.5Nm；反馈缺失/非有限也锁存阻尼。旧motor-test仍为Kp=5，未自动升级。修正VerifiedTransport的状态与反馈更新为一次锁内提交，保证日志中的q、dq和tau_est来自同一有效帧。

分析器区分估算力矩响应、编码器响应、两者均有响应、无法确认和提前停止；基线/正负窗口不足、有效tick不足或DDS回显CRC无效均不得宣称确认。力矩阈值为max(0.05Nm, 3倍基线稳健噪声)，角度阈值为max(0.001rad, 4倍基线角度跨度)，正负两个方向都需满足。实体腿的物理映射/方向始终保留为未确认，不能仅用SDK回显升级。具体命令与判据已并入真机指南。测试覆盖无响应、仅力矩、仅编码器、两者响应、数据不完整及在线反馈保护；软件和本机DDS结果在本节后续记录。

验证结果：38项软件测试通过，2项DDS集成默认跳过；单独启用本机DDS后，新增入口全部12项测试通过，其中真DDS使用lo/非零domain与模拟LowState，原厂RPC替换为测试对象，观察到唯一SDK髋索引3、Kp=20、正负目标分别达到±0.005弧度、正确CRC和独立停止后一秒阻尼。旧2秒日志离线分析输出inconclusive，不会误认其为完整3秒响应试验。未连接机器狗、未执行新增益的真机测试、未自动恢复原厂模式或启动策略。下一次需现场确认趴卧/停止终端，并明确说明新候选Kp=20后，再配合采集一次完整记录。

### 2026-10-09 首次完整实体响应诊断：力矩反馈通过，角度响应未确认

使用者确认开机卧倒、网线连接、场地与停止终端就绪。只读复核normal、约500Hz状态、抽查CRC正确后，执行`go2_commission eno1 --response-test --confirm-prone --release-mode`，3秒基线/正向/基线/负向/基线，FL_hip（SDK索引3）Kp=20、±0.005弧度；其余Kp=0。记录`logs/go2_real/commission/go2_real_20261009_190713_1791544033427066537.jsonl`，1040帧MotorTest、419帧到期阻尼，无安全故障。实际测试记录均值346.39Hz，最大状态年龄6.66毫秒，1458行含DDS回显且CRC均正确，电机lost为0。

自动分析结果`force_response_only`：正负峰值估算力矩相对基线中位数变化分别+0.07421Nm、-0.09895Nm，超过0.05Nm判据；角度变化分别+0.000424rad、-0.000151rad，低于本轮噪声自适应阈值0.002180rad（基线角度跨度0.000545rad）。全程任一关节相对起点最大变化0.000757rad，FL_hip反馈估算力矩绝对值最大0.17317Nm。使用者确认无异常动作或声响。

这个结果确认了与正负目标方向一致的电机估算力矩反馈变化，支持低层命令产生了控制作用；不是精确实测力矩或协议接收ACK，也未确认可辨认的角度跟踪/实体腿方向。地面接触、摩擦或当前激励幅度不足仍是候选解释，不能仅据本轮断言唯一原因。本轮未增加参数、未启动自定义站起或策略。

在使用者确认准备配合后，执行`--restore-normal --confirm-prone --allow-factory-stand`，查询确认normal恢复，告知可能原厂自动站起。使用者随后确认Go页面正常、已用App卧倒，收尾完成。当前原厂控制正常、测试进程结束，下一阶段仍需明确编码器响应与映射证据。

### 2026-10-09 逐关节验收准备与首次明确编码器响应

使用者要求集中完成部署前准备，避免反复开关机，并授权更易辨认的受限关节运动。确认机器人仍开机、原厂normal正常且卧倒。新增`--joint-check`：3秒、0.02弧度、仅所选关节Kp40，其余Kp0、Kd0.5；估算PD上限1Nm、反馈估算力矩2Nm、全关节实际位移0.03rad、速度0.5rad/s。支持全部12关节，按官方SDK关节次序校验映射。小腿起点低于模型下限时，只允许从当前实测起点向模型范围方向两次正向往返，禁止更折叠目标；起点偏差超过0.15rad或实际更折叠超过0.003rad则拒绝/阻尼，不改变策略限位。这只是受限测试，尚未解决策略起始姿态与模型范围差异。

新增测试覆盖12关节唯一驱动、固定参数、反馈与PD故障锁存、小腿方向与返回、分析器选定索引。43项发现测试中41项通过、2项默认跳过；单独启用本机DDS，新增入口全部15项通过，本次实际DDS选用RR小腿索引8，验证Kp40、目标只向内、CRC及独立停止后一秒阻尼。DDS使用lo和非零domain，原厂RPC替换为测试对象，无真机参与。分析器噪声基线修为初始0.4秒，不把激励后接触/滞后混入静止噪声；峰值窗口随目标幅度调整。

执行左前髋`--joint-check`，记录`logs/go2_real/commission/go2_real_20261009_192310_1791544990395069028.jsonl`。1015帧MotorTest、409帧阻尼，正常到期，无安全故障。分析结果force_and_encoder_response：正负角度相对基线变化+0.007993/-0.004844rad（约+0.46°/-0.28°），阈值0.001rad；估算力矩变化+0.4700/-0.5937Nm，阈值0.05Nm。所选髋最大位移0.008266rad，估算力矩绝对值最大0.6927Nm，全部关节最大速度0.1511rad/s。左前大腿/小腿被动变化最大约0.0020/0.00130rad，其余腿更小；没有其他关节正Kp。证据确认SDK索引3编码器实际正负响应，不等于实体映射或行走验收。

此前启动前未等待使用者确认正在观察。使用者先回答未看出动作、无异常，随后明确当时没有看机器人，且实际幅度肉眼难辨。已明确纠正：不能把此答复计为现场观察通过或全程无异常确认。下一次必须先告知所选关节与开始时间、等待观察就绪；微小角度用编码器量化。当前进程结束，已阻尼，原厂模式仍为空，未自动恢复normal，未启动站起或策略。App Go暂不可用与原厂模式退出一致，收尾恢复须协调可能自动站起及App卧倒。

### 2026-10-09 右前髋响应验收

使用者确认正在观察，启动前明确告知右前髋、0.02rad、Kp40、3秒及第二终端停止路径。执行`--joint-check --test-joint FR_hip_joint`，记录`logs/go2_real/commission/go2_real_20261009_193028_1791545428038216214.jsonl`。1051帧MotorTest、414帧阻尼，正常到期，无软件安全故障；正Kp仅SDK索引0。相对起点最大角度变化+0.004935/-0.000969rad，所选估算力矩绝对值最大0.8411Nm，全关节最大速度0.1511rad/s。

分析结果force_response_only：正负峰值窗口角度相对首段基线变化+0.004481/-0.000908rad（约+0.257°/-0.052°），负向低于预设0.001rad门槛；估算力矩变化+0.61846/-0.74215Nm，均超过0.05Nm。确认正向编码器响应及双向估算力矩响应，不能将双向角度判为通过，也不能据此确定接触/摩擦是唯一原因。未改变判据，未继续增加参数；暂停原定余下两髋顺序以核对结果。

使用者确认一直趴卧、无异常动作或声响。原厂模式仍为空，尚未执行恢复normal，App Go可能暂不可用；未执行自定义站立、保持或策略。当前已采集两髋，其余10关节未采集。使用者随后表示可主要依靠数据检测，不再要求肉眼辨认微小角度；现场观察仍用于异常与收尾。独立关节的未通过项目应单独保留，不能以其他项目结果替代或自动升级到站起/策略。

### 2026-10-09 剩余10关节一次连续采集完成

使用者要求完成全部前置工作，不再逐项问询，并保持现场观察与停止终端。明确告知连续采集方案后，按相同0.02rad/Kp40/3秒参数依次执行剩余10个关节，每项独立进程、正常到期后一秒阻尼，逐项检查退出码与日志故障；期间未恢复原厂模式、未关机、未运行策略。全部正常结束，没有软件安全故障。右前髋及前小腿响应不足独立保留，未改判据或自动加参数。

| 关节 | 角度变化（rad，两个峰值窗口） | 结果 |
| --- | --- | --- |
| FL_hip | +0.007993 / -0.004844 | 力矩与编码器响应 |
| FR_hip | +0.004481 / -0.000908 | 力矩响应；负向角度不足 |
| RL_hip | +0.007236 / -0.001786 | 力矩与编码器响应 |
| RR_hip | +0.001726 / -0.005405 | 力矩与编码器响应 |
| FL_thigh | +0.002816 / -0.001272 | 力矩与编码器响应 |
| FR_thigh | +0.002634 / -0.003452 | 力矩与编码器响应 |
| RL_thigh | +0.006782 / -0.002452 | 力矩与编码器响应 |
| RR_thigh | +0.002695 / -0.004360 | 力矩与编码器响应 |
| FL_calf | +0.000545 / +0.000584 | 力矩响应；角度不足 |
| FR_calf | +0.000569 / +0.000569 | 力矩响应；角度不足 |
| RL_calf | +0.001674 / +0.001690 | 力矩与编码器响应、返回起点 |
| RR_calf | +0.002101 / +0.002085 | 力矩与编码器响应、返回起点 |

后小腿为两次向减少折叠方向的脉冲；不能将其写成正负双向。12项均有对应方向估算力矩响应，9项符合本轮编码器判据。其余10项日志为`logs/go2_real/commission/go2_real_20261009_193436_*`至`193559_*`，汇总`joint_response_summary_20261009.json`。自动报告不确认实体腿映射，现场微小角度观察不作必需量化手段。整机承重站立及实际策略仍未验收，App原厂模式尚未恢复。

### 2026-10-09 无策略站立完整流程准备（尚未执行真机）

在同一`go2_commission.py`新增`--stand-check --allow-body-movement`，不加载策略。初始必须已趴卧；17秒分为3秒平滑进入官方卧倒目标、3秒官方站姿、2秒训练默认站姿、2秒保持、5秒降低回官方卧倒目标、2秒稳定，随后一秒阻尼。SDK目标取自本地官方`go2_stand_example.py`；起身/降低使用官方60/5，保持平滑切换到配置FixStand 40/1，未声称训练增益20/0.5已经验收。

前3秒仅允许实测深卧倒小腿向内进入模型范围，不扩大策略限位；之后按原模型范围检查。站立阶段姿态0.35rad、速度2rad/s、跟踪误差0.25rad、PD/反馈估算力矩12Nm、循环50ms、状态100ms与心跳100ms限制，故障锁存阻尼。末一秒要求至少50个新tick、连续0.8秒保持误差不超过0.2rad、姿态不超过0.2rad；正常结束还需趴卧与目标距离0.15rad检查。停止发生在站立时可能降低机身，不能承诺保持站立或完整链路断开后停机。

仅电脑端迭代：起身初稿40/1及8Nm上限在MuJoCo触发力矩/跟踪保护，后采用官方60/5和12Nm（低于既有最小策略力矩检查18.96Nm），保留0.25rad跟踪及其他保护。固定默认目标直接切换20/0.5在仿真不能满足保持跟踪，实际策略会输出动态目标，因此无策略保持改为配置FixStand 40/1，不将20/0.5计为已通过。0.15rad保持误差初稿在40/1稳定承重时仅29帧符合（最大约0.1613rad）；正式候选0.2rad连续0.8秒，仍小于0.25rad硬保护。所有候选调整均尚未用于真机，未更改已执行的关节验收参数或原策略限位。

完整MuJoCo流程正常到期：初始高度约0.082m、保持约0.308m、结束约0.091m，峰值PD约8.87Nm，500个新tick保持，降姿完成。产物`logs/go2_real/commission/stand_check_mujoco_result.json`；临时验证脚本`/tmp/go2_stand_check_sim.py`，不进入项目接口。46项默认测试运行时44项通过、2项跳过，新增完整阶段/无策略、状态故障、假重复tick和显式整机运动参数保护。后追加连续保持/保持增益检查需复测。

本机DDS完整流程两次触发循环50ms保护（后一轮66.6ms），已阻尼，未扩大循环阈值。它们使用本机lo/非零domain和模拟状态、原厂RPC替换，无真机参与。正在验证有界验收循环的垃圾回收处理：改变原厂模式前清理，在限定控制阶段暂停循环垃圾自动收集、退出恢复；引用计数仍释放非循环对象。尚未根据此修改宣称DDS通过，也未执行真机站立或策略。

随后复测通过：`GO2_DDS_TESTS=1 GO2_STAND_CHECK_TEST=1 ... unittest discover -s tests -p test_go2_commission.py`，18项全部通过，耗时26.46秒；完整五阶段、官方60/5到配置40/1保持、正确CRC、第二终端停止后一秒阻尼及未运行策略均验证。50ms限制未修改。修改与通过支持垃圾回收调整有助于消除该测试长暂停，但不证明所有调度迟延都来自垃圾回收或真机延迟保证。已向使用者明确17秒整机站起/保持/降低及中途阻尼可能降低机身，等待现场仍卧倒、无人、正在观察和停止终端就绪确认，尚未执行真机。

### 2026-10-09 整机趴卧起身前归位失败与原厂站姿复核

使用者确认整机测试全部就绪后，执行17秒候选。记录`go2_real_20261009_195830_1791547110025830983.jsonl`：978帧FixStand，在3.0029秒刚切换起身阶段时，四个小腿仍约-2.78至-2.80rad，未进入模型范围，触发实际位置保护并阻尼一秒；未运行策略。大腿/髋已移动，小腿仅向内移动0.012至0.017rad，反馈估算力矩最大9.6727Nm。不能认定控制无效，也不能放宽模型限位直接进入策略。已告知使用者并未继续增加卧倒小腿力矩。

明确告知恢复normal可能自动站起、此次先保持站立后恢复原厂。使用者确认原厂已站稳、App Go正常。只读诊断`logs/go2_real/go2_readonly_diagnostic_20261009_200138.json`：normal、约500Hz、抽查CRC正确、lost0，原厂小腿-1.49至-1.54rad、姿态接近水平、SportModeState高度约0.32m，站姿在模型范围内。原厂小腿估算力矩约6.88至7.78Nm，证明本次承重不能用此前0.8Nm微动所需力矩估计。

新增`--stand-check --start-standing --confirm-standing --release-mode --allow-body-movement`，14秒原厂站姿接管。5秒实测站姿到训练站姿、2秒保持、5秒降低、2秒稳定，再一秒阻尼。发布器和日志在原厂仍保持站姿时准备，尚不发送；确认模式释放后立即保持，避免原来发布器构造的250ms等待。主终端在启动准备期也发送租约，释放前及首条保持前均检查新鲜租约/启动停止；主进程缺失或启动停止不能启动主动控制。释放后监测LowCmd是否匹配本程序最近128个指令签名及CRC，不匹配锁存阻尼；不承诺硬件完整断链后停机。

本机DDS21项通过，原厂RPC替换且模拟状态、lo/非零domain，验证启动租约、完整14秒阶段、消息冲突检测、CRC、独立停止后一秒阻尼及未运行策略；此后新增启动租约独立测试。50项默认发现测试48项通过、2项跳过。MuJoCo14秒正常完成，保持高度约0.314m，最低接管高度约0.293m，末端约0.096m，峰值PD约8.49Nm；未运行策略。

### 2026-10-09 首次实体站姿接管与保持成功，降低末端触限

明确告知从已确认原厂站姿启动14秒完整测试，记录`go2_real_20261009_201748_1791548268168691313.jsonl`。5889帧FixStand，完成5秒站姿过渡和2秒保持，达到在线保持判据；下降第11.8999秒FR小腿实测-2.7427106rad，略低于模型状态下限-2.7427，触发位置保护并阻尼一秒。反馈估算力矩最高7.8235Nm；没有模式冲突、通信、姿态、速度或力矩故障记录。整体仍判aborted，不将其写成全流程通过；站姿接管/保持已有独立实体证据。

使用者确认先平稳站立、随后降低、现在趴卧且无异常。原厂尚未恢复，未运行策略。问题集中于降低目标-2.65rad加负载误差使实际小腿触及模型下限。调整仅站姿起始流程的降低目标为-2.55rad（向更少折叠方向0.1rad），原模型上下限、50ms循环及其他保护未改。末端先验证低姿态与静止，再阻尼放到地面，不能以末端关节姿态声称已接触地面；日志完整判定还需阻尼后的趴卧检查与现场确认。

修正终点后MuJoCo14秒正常到期，保持约0.314m，主动降低末端约0.107m，峰值PD约8.49Nm，500个新tick保持；输出`standing_handoff_mujoco_result.json`。仅终点及低姿态检查变化，发布/租约/停止协议沿用已通过版本；软件回归进行中，尚未执行修正终点的真机复核或策略。

### 2026-10-09 完整站姿接管通过与首次真机策略试跑

恢复normal后立即接管曾因原厂尚未站稳而在只读预检退出，未释放模式或发布LowCmd。增加最长15秒等待，要求连续一秒站稳；随后14秒实体复核完整通过：`go2_real_20261009_202857_1791548937567603811.jsonl`，6949帧FixStand、501帧阻尼，保持464个新tick、最大误差0.12245rad，反馈估算力矩峰值7.7761Nm。未运行策略，阻尼后编码器符合趴卧检查。

使用者明确“开始”，授权首次2秒零速度策略。新增独立`--policy-check READY_JSONL`，必须先提供通过的无策略站立日志；加载/预热策略在DDS及模式变更前。5秒站姿接管、2秒物理保持判据通过后，运行2秒实际策略（速度固定零、Kp20/Kd0.5、20ms计算周期）；500Hz计划发布缓存目标。切换退出增益时按当前q/dq保持估算PD，再5秒降低、2秒稳定、一秒阻尼，不能延时或给行走速度。保留100ms租约/状态、50ms发布循环、原模型限位、0.35rad姿态、2rad/s关节速度及12Nm PD/反馈估算力矩限制。异常直接阻尼，不尝试故障后继续降低。

实际冻结策略在MuJoCo完整16秒流程正常到期，保持高度约0.314m，策略峰值PD10.038Nm，降低末端约0.108m；仿真报告`policy_trial_mujoco_result.json`。新增测试验证拒绝失败的前置日志、真实保持门槛、500Hz输出只进行50Hz策略计算、固定零速度及停止锁存。52项软件测试50通过、2项跳过；包含本机停止套接字测试，受限沙箱首次禁止套接字后在批准环境运行通过。

已明确告知现场原厂恢复可能站起、接管和策略开始时间，使用者保持观察和停止终端。实体日志`go2_real_20261009_204247_1791549767838902699.jsonl`：正常到期，无安全故障；6942帧FixStand、974帧Policy、502帧Damping，策略时间跨度1.99671秒、96次推理、879个独立状态tick，计划20ms周期因Python调度实际略低于50Hz。反馈估算力矩峰值11.56927Nm、策略PD峰值10.98996Nm；最后关节速度最大0.06201rad/s、编码器符合趴卧检查。分析结果`zero_velocity_policy_trial_passed`仅代表本轮日志判据通过，现场动作反馈待核对，未测试真机行走、键盘速度控制、长时运行或完整断链停机。

本轮结束后独立进程已退出，已发送一秒阻尼，原厂模式尚未恢复；等待现场确认动作/姿态再协调收尾。README与真机指南已同步入口、真实进度及限制。未提交或推送本轮更改。

### 2026-10-09 首次限时前进候选、接管修复及未迈步的原因

使用者确认上一轮稳定、无异常且已趴卧，并再次确认前方至少1米平坦空地、持续观察与停止终端，授权最高0.10m/s、2秒前进与2秒归零观察。新增`--forward-check ZERO_JSONL`，前置零速日志必须通过，固定19秒：5秒接管、2秒保持、1秒零速策略、2秒前进（前后0.3秒平滑加减速）、2秒归零、5秒降低、2秒稳定，再一秒阻尼。策略增益与45维观测/12关节映射未改，继续50Hz计划推理、500Hz计划发布。

真实失败记录与修复逐项保留，不能当作前进通过：

- `go2_real_20261009_205723_1791550643082687472.jsonl`：3467帧FixStand、112帧Policy、500帧Damping，策略开始约0.229秒，仍为零速度，左后小腿反馈估算力矩12.04342Nm超出12Nm；前一帧PD11.524Nm。未发送前进指令。使用者确认无异常且已趴卧。
- 增加0.5秒平滑接管后`go2_real_20261009_210251_1791550971010777935.jsonl`仍于策略开始约0.451秒触发相同反馈保护：前一帧左后小腿PD11.519Nm，反馈12.04342Nm。只有平滑接管未解决负载峰值，现场再次确认无异常、已趴卧、继续观察。
- 对策略阶段增加10.5Nm PD输出限幅，通过当前q/dq将目标反算为`q+(clip(PD)+Kd*dq)/Kp`，并检查反算目标仍在原模型范围。12Nm反馈保护、姿态与通信保护未放宽。离线回放失败记录中41帧会被限幅，PD峰值由11.519降至10.5Nm；这本身不代表真实电机验证。
- `go2_real_20261009_210813_1791551293994228059.jsonl`：只发487帧Damping，没有正Kp/Policy。原因是原厂释放后仍重新要求严格静止起点，释放后dq约0.52rad/s使起点检查失败。已修复为在原厂仍承重时严格验证并记录起点，再释放并立即按主动控制条件保持；若释放未确认，不在finally发送与原厂竞争的LowCmd。使用者确认趴卧、无异常。

停止后自然展开姿态不完全符合先前逐关节测试的趴卧范围，因此仅恢复normal采用独立`check_resting_for_restore`：状态新鲜、姿态≤0.2rad、dq≤0.2rad/s、髋≤0.7rad、大腿0.8至1.6、小腿-2.9至-2.4rad，并要求现场确认接触地面。该检查不适用于主动受限电机测试，旧`check_prone`及策略模型限位未修改。只读诊断`go2_readonly_diagnostic_20261009_205958.json`确认原厂释放、CRC正确、约500Hz、lost0、没有其他LowCmd发布、dq约0.058rad/s。

软件验证：默认Go2回归先56项54通过/2跳过；接管时序修复后本机lo/独立domain、真实SDK及模拟原厂RPC的28项全部通过，其中模拟释放后dq=0.4rad/s持续50ms，验证严格检查在释放前完成、第一条主动保持及独立停止。新增前进门槛禁止用未验证当前0.5秒接管/10.5Nm限幅的旧零速日志。后续默认58项56通过/2跳过；最终速度候选修改后补充的两项测试通过，分别覆盖0.10/0.30速度完整时序、归零、退出、日志判据，以及策略/站立速度限制与超限停止。

修复后的真机复核通过：`go2_real_20261009_211242_1791551562254476331.jsonl`，6831帧FixStand、969帧Policy、502帧Damping；策略跨度1.99870秒，96次推理，反馈估算力矩峰值11.09512Nm、策略PD峰值10.500002Nm，正常降低与阻尼，末态编码器符合趴卧检查。随后明确告知启动时间，用这份新日志作为前置条件执行已授权的0.10m/s前进测试。

`go2_real_20261009_211608_1791551768057560291.jsonl`：6849帧FixStand、2406帧Policy、498帧Damping，策略跨度4.99776秒、238次推理，确实下发最高0.10000000149m/s前进指令并按时归零，反馈估算力矩峰值11.18995Nm，无故障、正常降低与阻尼。分析仅`forward_policy_trial_log_passed`，实体平移与停走标志为false。使用者明确反馈“没有向前迈步”，因此不能计为真机行走通过。

选速问题：`legged_robot.py`的训练指令采样会将平面速度幅值≤0.2m/s置零；0.10m/s不适合正常迈步验收。冻结策略的MuJoCo候选：0.21m/s仅约0.022m前移；0.25m/s在归零阶段触发2rad/s限速；0.30m/s正常摆腿先触发2与4rad/s候选限速。仅在仿真诊断使用现有策略配置25rad/s观察完整运动，得到摆腿速度峰值6.16018rad/s、策略阶段前移0.19732m，归零后0.5秒起速度峰值0.02067m/s，完整降低正常到期；不能把这些当作真机效果。

最终0.30m/s候选仅策略阶段使用8rad/s，零速度模式、0.10m/s对照及站起/降低继续2rad/s；保留PD10.5Nm、反馈12Nm、姿态0.35rad、原模型位置范围与租约/状态/循环保护。URDF髋/大腿速度上限30.1rad/s，小腿20.07rad/s；模型数值不是候选真机稳定性保证。最终8rad/s候选MuJoCo完整19秒正常到期，最大dq6.16018rad/s，约0.197m前移，末端高度约0.106m。保存`forward_candidate_03_mujoco_result.json`（该文件保留最终候选结果）。新增显式`--forward-speed 0.3`；默认仍0.1，避免旧命令意外增速。

0.30m/s及8rad/s策略候选尚未执行真机。已经同步README和真机指南，下一步需现场另行确认较大速度这一轮。当前独立验收进程已退出、阻尼已发送，原厂模式保持释放，不自动恢复或站起。未提交或推送本轮更改。

### 2026-10-09 首次0.30m/s短时迈步与停走现场确认

使用者另行“确认”0.30m/s候选、前方空地、观察及停止终端。核对当前程序、成功零速前置日志和最终MuJoCo报告（实际文件为`forward_candidate_03_mujoco_result.json`，已修正上一节路径记录），明确告知恢复normal可能站起，随后运行`--forward-check ...211242...jsonl --forward-speed 0.3 --start-standing --confirm-standing --release-mode --allow-body-movement --allow-policy-movement`。本轮未修改速度、力矩、姿态或位置阈值。

实体日志`go2_real_20261009_214812_1791553692254684605.jsonl`：6838帧FixStand、2395帧Policy、499帧Damping，正常到期且无安全故障；最高前进目标0.3000000119m/s，前进阶段跨度1.99877秒，10.1秒起归零观察跨度1.89926秒（完整零速阶段约2秒），之后完整降低与稳定并发送一秒阻尼。反馈估算力矩峰值11.09512Nm、策略PD峰值10.500002Nm、策略关节速度峰值1.35256rad/s、策略横滚/俯仰绝对峰值0.05750rad。末态dq最大0.05813rad/s，编码器符合趴卧检查。

使用者明确反馈“向前迈步，随后停止并趴卧，无异常”。此为首次真实短时策略迈步、归零停走和降低的现场证据；没有独立位姿测量，不能报告真实距离、精确速度或误差。未验证键盘真机方向输入、侧移转向、持续行走、组合指令或完整断链停机。将现场观察写入独立汇总`forward_trial_03_summary_20261009.json`，不改原始DDS日志。

日志自动分析仍为inconclusive，并非力矩/姿态故障：起点在释放原厂服务前严格验证时已开始计时，此次RPC/初始化使首条主动日志的elapsed为0.239997秒，最后18.99819秒，主动记录跨度18.75820秒，低于18.8秒全控制时长判据。后续保持、5秒策略（含前进和归零）、5秒降低、2秒稳定均完整；不能将全控制时长判据写为通过，原始记录及18.8秒判据保留。分析消息改为准确描述“完整控制时长记录不足；需核对接管计时起点”。

修复时序：新增`begin_handoff`，释放前仍严格验证和保存原厂承重起点；释放确认及新鲜租约之后，再开始完整运动时钟，同时更新策略FSM的FixStand起始时钟。初始目标与状态/运动保护不变，不能重置已运行、停止或重复接管的时钟；日志记录释放准备延迟。新增单元测试模拟0.25秒释放延迟、释放后dq=0.4rad/s，验证保留起点、完整从零计时及不能再次重置；结合0.10/0.30前进完整时间曲线测试两项通过。此项软件修复未重复真机测试。

README、真机指南和工作日志已同步真实现场结果及未通过的全时长分析判据。当前试跑进程已退出、已阻尼、使用者确认趴卧；原厂模式尚未恢复，没有再次发送主动命令。未提交或推送本轮更改。

### 2026-10-09 GitHub同步前检查

使用者要求“更新github”。远端`origin/main`与本地提交基线均为`8ff1025`。本次同步范围为真机验收入口`go2_commission.py`、其测试、记录器反馈扩展、第二终端停止接口、README、真机指南与本工作日志，共7个文件。已有的MuJoCo注释、训练代码空白及图片删除改动保留在工作区，未纳入本轮部署提交；本机原始采集日志依照现有gitignore保留本地。

提交前运行完整Go2回归：59项，57通过、2项本机DDS集成测试跳过；独立停止/心跳丢失套接字场景通过。`git diff --check`通过。接管时钟修复仅软件验证，真实短时行走/停走现场证据与全时长日志判据的限制仍明确保留，未将未验证项写为通过。
