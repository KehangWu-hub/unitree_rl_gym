# Unitree Go2 RL Gym

面向 Unitree Go2 的强化学习运动控制项目，包含 Isaac Gym 训练、MuJoCo 仿真验证和基于 Unitree SDK2 / DDS 的真机部署工具。

项目支持原有 PPO 策略和 HIM 策略。各策略的观测顺序、历史和控制参数随 TorchScript 一起导出，MuJoCo 与真机共用部署接口。键盘控制支持前进、后退、左右侧移、行进中转向和速度归零。

[环境安装](doc/setup_zh.md) · [真机部署](deploy/deploy_real/README_GO2.zh.md) · [HIM部署与验收](deploy/deploy_real/README_GO2_HIM.zh.md)

| Isaac Gym | MuJoCo | 真机 |
|:---:|:---:|:---:|
| <img src="doc/images/go2-isaacgym.png" alt="本项目 Go2 策略在 Isaac Gym 中的回放" width="240" height="180"> | <img src="doc/images/go2-mujoco-preview.png" alt="Go2 在 MuJoCo 中的仿真画面" width="240" height="180"> | <img src="doc/images/go2-real-preview.jpg" alt="Go2 实体机器人" width="240" height="180"> |

## 特性

- **两种训练结构**：原 PPO 使用45维Actor观测；HIM使用六帧历史估计3维速度和16维隐变量，再输入64维Actor。Critic额外信息仅用于训练。
- **一致的策略接口**：训练导出观测顺序、缩放、关节映射和控制参数，MuJoCo 与真机共用观测构造及动作转换。
- **完整的仿真路径**：支持 Python MuJoCo、官方 DDS 仿真闭环、速度指令和外力扰动测试。
- **受限真机控制**：提供站姿接管、键盘运动、速度归零、受控降低、阻尼停止及运行日志。

## 工作流程

```text
Isaac Gym / PPO 或 HIM
    │  导出完整推理模型 + deploy.yaml
    ▼
TorchScript 策略
    ├── MuJoCo：运动与稳定性验证
    ├── DDS 仿真：通信与停止机制验证
    └── Go2：只读检查 → 站姿接管 → 受限运动
```

| 策略参数 | 原 PPO（`go2`） | HIM（`go2_him`） |
|---|---|---|
| 部署模型输入 | 45维单帧观测 | 270维历史（6 × 45，最新帧在前） |
| Actor输入 | 45维 | 64维＝当前观测45＋估计速度3＋隐变量16 |
| Critic输入（仅训练） | 60维 | 238维 |
| 动作输出 | 12个关节位置动作 | 12个关节位置动作 |
| 策略周期 | 20ms（50Hz） | 20ms（50Hz） |
| 动作缩放 | 0.25 | 0.25 |
| 策略PD增益 | Kp＝20，Kd＝0.5 | Kp＝20，Kd＝0.5 |
| 预训练策略 | `deploy/pre_train/go2/motion.pt` | `deploy/pre_train/go2_him/policy.pt` |

HIM结构如下。部署时只保留历史编码器和Actor；Target Encoder、对比学习和Critic参与训练。
图中隐式状态在当前配置中为16维，另有3维显式速度估计。

![HIM混合内部模型与策略训练结构](doc/images/HIM.png)

HIM模型接口与权重替换方法见 [模型说明](deploy/pre_train/go2_him/README.md)。
本地继续训练的候选模型须通过同一套仿真验收，才能替换部署权重。

## 目录用途

| 目录 | 用途 |
|---|---|
| `legged_gym/` | Isaac Gym 训练、策略回放和导出 |
| `deploy/deploy_mujoco/` | 普通 MuJoCo 运行与键盘流程验证 |
| `deploy/deploy_real/` | 真机连接、控制和停止 |
| `deploy/common/` | 策略接口，以及保留供研究复用的速度反馈实现 |
| `tests/` | 自动检查代码、限制与停止逻辑；日常运行不需要打开 |
| `logs/` | 本地训练记录、实验数据和图表，不随 Git 上传 |

## 开始使用

```bash
git clone https://github.com/KehangWu-hub/unitree_rl_gym.git
cd unitree_rl_gym
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

依赖安装见[环境安装说明](doc/setup_zh.md)。下文统一使用 `unitree-rl` Conda 环境，训练、仿真和真机部署的依赖均安装在该环境中。训练需要 Isaac Gym、NVIDIA GPU 和 `rsl_rl`；MuJoCo 仿真需要 PyTorch、NumPy、PyYAML 和 MuJoCo，真机 DDS 通信另需 `unitree_sdk2py`。仅使用已有策略进行仿真或真机部署时，无需安装 Isaac Gym 和 `rsl_rl`。

仓库提供预训练策略和真机配置。安装部署依赖后，可先检查模型，无需连接机器人：

```bash
conda activate unitree-rl
python deploy/deploy_real/deploy_real_go2.py --check
```

### 训练与导出

安装训练依赖后运行：

```bash
conda activate unitree-rl
python legged_gym/scripts/train.py \
  --task go2 --headless --num_envs 4096 --max_iterations 1500 \
  --run_name go2_45x60_official_rewards
```

使用训练记录的实际目录名进行 Play 和导出：

```bash
python legged_gym/scripts/play.py \
  --task go2 --headless --load_run RUN_DIRECTORY --checkpoint 1500
```

默认导出目录：

```text
logs/rough_go2_45x60_rewards/exported/
├── policies/policy_1.pt
└── params/deploy.yaml
```

策略与 `deploy.yaml` 必须来自同一次导出。训练记录与导出目录不随 Git 分发；正式部署前应完成对应仿真验证。

#### HIM训练任务

`go2_him` 使用 Isaac Gym 和项目内固定版本的 HIMLoco 算法，输入为六帧45维观测；
历史编码器估计速度与隐变量，和当前观测一起输入Actor。无需替换已安装的 `rsl_rl`。

```bash
python legged_gym/scripts/train.py --task go2_him --headless --num_envs 64
python legged_gym/scripts/train.py --task go2_him --headless --num_envs 64 \
  --resume --load_run RUN_DIRECTORY --checkpoint CHECKPOINT
python legged_gym/scripts/play.py --task go2_him --headless --num_envs 16 \
  --load_run RUN_DIRECTORY --checkpoint CHECKPOINT --export_only
```

环境数量可按显存调整。导出文件为 `logs/go2_him/exported/policies/policy.pt` 和
`logs/go2_him/exported/params/deploy.yaml`。HIM导出包含编码器和Actor，参数文件使用
时间序列接口版本2；部署时选择独立的 `go2_him.yaml`。原 `go2` 任务和真机模型路径保持独立。
HIM仿真复现、离线检查和现场验收命令见 [HIM部署指南](deploy/deploy_real/README_GO2_HIM.zh.md)。

### MuJoCo 仿真

默认仿真配置使用上面的导出文件。准备好策略与参数后运行：

```bash
conda activate unitree-rl
python deploy/deploy_mujoco/deploy_go2.py --command 0.5 0.0 0.0
```

`--command` 依次为前进速度、侧向速度和偏航角速度，单位为 m/s、m/s、rad/s。无窗口运行及扰动测试：

```bash
python deploy/deploy_mujoco/deploy_go2.py \
  --headless --no-realtime --duration 10 --command 0.5 0.0 0.0

python deploy/deploy_mujoco/deploy_go2.py \
  --headless --no-realtime --duration 10 --command 0.5 0.0 0.0 \
  --push-at 5 --push-duration 0.2 --push-force 50 0 0
```

程序输出位移、速度误差、机身高度、倾角和力矩等指标。模型路径与物理参数在 [MuJoCo 配置](deploy/deploy_mujoco/configs/go2.yaml) 中设置，也可通过 `--policy` 和 `--deploy-config` 指定成套导出文件。

速度统计由上述仿真入口直接输出：`mean_body_velocity` 为机身坐标系平均速度，`velocity_error_xy` 为线速度误差。保留的 PI 补偿实现及参数见[速度反馈运行说明](deploy/deploy_real/README_GO2.zh.md)；普通仿真入口不会自动启用这些补偿。

### 真机部署

真机控制使用 `go2_keyboard`，默认统一接收 Q/W/E/A/S/D，X 将速度归零，也保留单方向检查模式。从原厂站稳接管后，统一模式默认持续运行，C 受控降低并退出，Z/Ctrl+C 或第二终端停止紧急阻尼退出；可用 `--motion-seconds 4` 或 `10` 选择自动归零和降低的短时检查。运动入口默认对合法动作做步长限幅。默认 W/S 为 ±0.50m/s、A/D 为 ±0.40m/s；Q/E 为前进 0.50m/s 加转向 ±0.50rad/s。这些是策略指令，实际速度取决于跟踪效果。

历史单独 A 速度对照可显式使用 `--speed-profile legacy --walk-speed 0.5 --left-speed 0.4`，D保持0.30，见[速度对照步骤](deploy/deploy_real/README_GO2.zh.md#单独对照左移速度)。

左移起步恢复候选可用 `--gait-restart`：检测到左移步态持续静止后，平滑回站姿并重新起步一次。使用与停止方法见[真机指南](deploy/deploy_real/README_GO2.zh.md#左移起步恢复候选)。

原厂速度消息在当前低层接管后不可用。已接入关节与IMU运动学速度估计，首轮用 `--velocity-source kinematic --velocity-feedback shadow` 记录估计，保持普通控制输出；核对真机速度量级后再启用PI。可选仅 PI 候选的反馈检查、低速参数和运行步骤见[速度反馈运行说明](deploy/deploy_real/README_GO2.zh.md)。

部署需要匹配当前模型与配置的站立、零速度前置记录；这些记录保存在本机，克隆仓库后需自行完成检查。连接、前置流程、运行命令及停止方法见 **[Go2 真机部署指南](deploy/deploy_real/README_GO2.zh.md)**。

已激活 `unitree-rl` 环境且机器人原厂站稳时，运行：

```bash
python -m deploy.deploy_real.go2_keyboard --run
```

| 按键 | 功能 |
|---|---|
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移 |
| Q / E | 行进中左转 / 右转 |
| X / 空格 | 停步，保持策略运行，可继续运动 |
| C | 正常趴下并退出；运动后制动 2 秒、过渡站姿 2 秒、保持 4 秒、降低 8 秒、稳定 3 秒、阻尼收尾 1 秒 |
| Z / Ctrl+C | 紧急终止，立即进入阻尼并退出 |

C 用于默认持续模式，按一次即可；退出过程中仍可按 Z 紧急终止。可选短时检查自动降低，方向切换无需先按 X。

## 测试

```bash
conda activate unitree-rl
python -m unittest discover -s tests -p 'test_go2*.py'
```

测试覆盖策略接口、模型与关节映射、状态机、保护逻辑、键盘输入和看门狗。默认跳过需要本地 DDS 网络的集成测试。启用 MuJoCo / DDS 停止机制测试：

```bash
GO2_DDS_TESTS=1 python -m unittest discover -s tests -p test_go2_watchdog.py -v
```

## 项目结构

```text
legged_gym/                 Isaac Gym 环境、训练和策略导出
  envs/go2/                原 PPO 配置与观测定义
  envs/go2_him/            HIM 环境、配置与完整模型导出
deploy/
  common/                  共用策略接口
  deploy_mujoco/           MuJoCo 运行与诊断
  deploy_real/             DDS 通信、真机控制与日志
  pre_train/go2/           原冻结 TorchScript 模型
  pre_train/go2_him/       HIM 模型、接口与版本记录
resources/robots/go2/      URDF、MJCF 与模型资产
tests/                     Go2 软件与集成测试
doc/                       安装与技术文档
```

本项目基于 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 和 [HIMLoco](https://github.com/InternRobotics/HIMLoco) 等开源实现开发，许可证见 [LICENSE](LICENSE)。
