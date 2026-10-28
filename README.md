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

| 策略参数 | 原 PPO（`go2`） | HIM（`go2_him_retrain`） |
|---|---|---|
| 部署模型输入 | 45维单帧观测 | 270维历史（6 × 45，最新帧在前） |
| Actor输入 | 45维 | 64维＝当前观测45＋估计速度3＋隐变量16 |
| Critic输入（仅训练） | 60维 | 238维 |
| 动作输出 | 12个关节位置动作 | 12个关节位置动作 |
| 策略周期 | 20ms（50Hz） | 20ms（50Hz） |
| 动作缩放 | 0.25 | 0.25 |
| 策略PD增益 | Kp＝20，Kd＝0.5 | Kp＝20，Kd＝0.5 |
| 预训练策略 | `deploy/pre_train/go2/motion.pt` | `deploy/pre_train/go2_him_fresh/policy.pt` |

HIM结构如下。部署时只保留历史编码器和Actor；Target Encoder、对比学习和Critic参与训练。
图中隐式状态在当前配置中为16维，另有3维显式速度估计。

![HIM混合内部模型与策略训练结构](doc/images/HIM.png)

HIM候选尚未完成可靠真机部署。v3在MuJoCo已出现停步失稳，随后实机翻车并持续输出位置控制，当前已封停v3真机入口。独立倾角保护已恢复，不受动作/力矩整形开关影响，触发后锁定零Kp阻尼；该修复仅完成离线验证，未再次操作真机。旧候选及失败证据保留，不能将接口或机身指标通过视为步态合格。

HIM模型接口与权重替换方法见 [模型说明](deploy/pre_train/go2_him/README.md)。
本地继续训练的候选模型须通过同一套仿真验收，才能替换部署权重。

## 架构与职责

| 目录 | 用途 |
|---|---|
| `legged_gym/` | Isaac Gym 训练、策略回放和导出 |
| `deploy/deploy_mujoco/` | MuJoCo运行、独立步态指标、断点评估和录像 |
| `deploy/deploy_real/` | 真机连接、控制和停止 |
| `deploy/common/` | 共用策略接口、速度估计、反馈控制与评估证据校验 |
| `tests/` | 自动检查代码、限制与停止逻辑；日常运行不需要打开 |
| `third_party/` | 固定版本的HIM算法实现 |
| `resources/` | Go2机器人模型与仿真资产 |
| `deploy/pre_train/` | 成套部署模型、参数与版本记录 |
| `logs/` | 本地训练记录、实验数据和图表，不随 Git 上传 |

策略接口负责观测历史和动作转换；训练与仿真工具通过该接口对齐模型约定。真机侧另包含DDS通信、状态机与独立停止保护。MuJoCo步态指标集中在 `gait_metrics.py`，比较与断点监控入口复用同一实现。

模型版本由权重、`deploy.yaml` 和记录摘要共同标识。历史实验保留在 `logs/`，软件流程通过与步态质量通过分别记录。

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

从零训练入口为 `legged_gym/scripts/train_him_from_scratch.py`，断点续训为 `continue_him_training.py`。训练profile区分平地基线与混合地形候选，参数以各运行目录中的 `env.yaml`、`train.yaml` 和源码快照为准。

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

真机入口为 `go2_keyboard`，具体接管、配置选择和停止流程见[Go2部署指南](deploy/deploy_real/README_GO2.zh.md)及[HIM部署指南](deploy/deploy_real/README_GO2_HIM.zh.md)。模型切换后需要重新核对对应的仿真证据和现场记录。

| 默认键盘模式 | W / S | A / D | Q / E |
|---|---|---|---|
| PPO统一模式 | 前后±0.50m/s | 左右±0.40m/s | 前进0.50m/s＋转向±0.50rad/s |
| HIM历史模式 | 前后±0.30m/s | 左右±0.30m/s | 前进0.30m/s＋转向±0.30rad/s |

表中是策略指令，实际速度取决于跟踪效果。默认命令选择PPO配置；HIM需显式指定匹配的模型配置和记录。独立倾角停止保护不随动作整形开关关闭。速度估计、PI反馈及起步恢复等研究选项见部署指南。

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

## 开发与检查

在已安装依赖的环境中，从项目根目录运行：

```bash
make help
make test       # 全部离线测试，关闭可选DDS集成测试
make check      # 检查补丁空白错误
make clean      # 清除源码目录中的Python缓存
```

可用 `make test PYTHON=/path/to/python` 指定解释器。测试覆盖策略接口、训练约定、MuJoCo模型、状态机、速度估计和停止保护；部分测试需要本地回环socket权限。`make`入口不会连接或控制机器人。

可选DDS集成测试需单独启用，运行环境要求见部署指南：

```bash
GO2_DDS_TESTS=1 python -m unittest discover -s tests -p test_go2_watchdog.py -v
```

本项目基于 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 和 [HIMLoco](https://github.com/InternRobotics/HIMLoco) 等开源实现开发，许可证见 [LICENSE](LICENSE)。
