# Go2 真机部署指南

本文介绍 Go2 策略部署的有线连接、前置检查、键盘控制和停止操作。所有命令均从项目根目录运行，使用项目的 `unitree-rl` Conda 环境，并确保已安装 `unitree_sdk2py`。

[项目首页](../../README.md) · [环境安装](../../doc/setup_zh.md)

## 运行前准备

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

机器人应处于平坦空地，网线保持松弛，有人在旁观察，并准备独立的停止终端。每次主动控制前确认当前姿态；键盘运动从原厂站稳开始，不从卧倒姿态直接启动。

真机使用有线网卡和 DDS domain 0。下文以 `eno1` 为例，请替换成实际网卡。测试进程不可与其他自定义 LowCmd 发布程序同时运行。

控制器接管后会释放原厂运动服务，手机 App 的运动控制可能暂时不可用。主动停止使用键盘或下文的停止终端。软件停止仍依赖电脑、进程和通信，不能覆盖电脑断电或完整物理断链。

## 检查模型和连接

先检查配置与策略模型，不初始化 DDS：

```bash
python deploy/deploy_real/deploy_real_go2.py --check
```

只读取机器人状态，不创建电机发布器：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 --read-only --duration 12
```

原厂已站稳时，可用真实状态计算策略目标，仍不下发命令：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 \
  --read-only --shadow-policy --duration 12 --command 0 0 0
```

`deploy_real_go2.py` 的实体网卡入口仅允许只读操作。主动真机流程由 `go2_keyboard` 和 `go2_commission` 执行。

## 首次建立前置记录

`go2_keyboard` 会核对模型、配置和前置日志。[基线清单](configs/go2_verified_baseline.json) 保存模型、配置和前置记录的摘要；运行日志保存在本地，不随仓库分发。新环境需依次完成站姿接管和零速度策略检查，再将产生的实际日志路径传入后续命令。

### 1. 站姿接管

先由原厂模式让机器人站稳，确认现场和停止终端就绪：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --stand-check --start-standing --confirm-standing \
  --release-mode --allow-body-movement
```

程序等待连续一秒站稳后接管：5 秒过渡至训练站姿、2 秒保持、5 秒降低、2 秒稳定，随后发送 1 秒阻尼。不运行策略，不自动恢复原厂模式。

检查现场结果，并分析程序输出的日志文件：

```bash
python -m deploy.deploy_real.go2_commission --analyze STAND_JSONL
```

将 `STAND_JSONL` 替换为本轮实际路径。日志与现场均通过后，再进行下一步。

### 2. 零速度策略

按“恢复原厂模式”一节重新站稳，然后使用站立记录：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode zero --readiness STAND_JSONL --check

python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode zero --readiness STAND_JSONL --run
```

在现场交互终端输入 `START`。该流程包含 5 秒接管、2 秒保持、2 秒零速度策略、5 秒降低、2 秒稳定及 1 秒阻尼。零速度目标仍可能引起迈步或漂移；此模式下 X、空格、Z 和 Ctrl+C 都进入阻尼停止。

分析生成的 `ZERO_JSONL`，并现场确认停止与收尾。运动流程使用这份零速度记录，要求其模型、配置、接管曲线和力矩参数匹配当前版本。

## 键盘运动

每轮先恢复原厂站稳，保留观察人员及停止终端。已有匹配前置记录的本机可省略 `--readiness`；新环境使用自己的 `ZERO_JSONL`。

### 统一控制入口

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --readiness ZERO_JSONL --check

python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --readiness ZERO_JSONL --run
```

默认 `all` 模式在同一轮接受 Q/W/E/A/S/D，无需重新启动来切换方向。W/S 前进或后退，A/D 左右侧移，平移指令上限均为 0.30m/s。Q/E 向前移动并叠加左右转向，上限为 0.30m/s 和 ±0.30rad/s。

每次只执行最后收到的方向键，不组合多个按键。方向改变时先归零一个策略周期，再向新方向加速。X/空格将所有速度分量归零；从转向切回平移时不会保留转向指令。

统一模式共用一份 4 秒运动预算，切换方向或按 X 不会重置预算。预算耗尽后，方向键不再产生运动指令。

### 单方向模式

可用以下模式限制允许输入的方向键，前置记录与统一模式相同：

| 模式 | 方向键 | 指令 |
|---|---|---|
| `--mode walk-turn` | Q/E | 前进 0.30m/s，加左右转向 0.30rad/s |
| `--mode backward` | S | 后退 0.30m/s |
| `--mode lateral` | A/D | 左右侧移 0.30m/s |

例如仅启用侧移：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode lateral --readiness ZERO_JSONL --run
```

### 输入与时序

统一模式接收全部方向键；单方向模式仅接收表中对应方向键。

| 输入 | 作用 |
|---|---|
| Q / E | 行进中左转 / 右转 |
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移 |
| X / 空格 | 速度指令立即归零，策略继续运行 |
| Z / Ctrl+C | 终止流程并进入阻尼 |

输入 `START` 后暂不操作 App。看到方向键窗口提示再按键，主动停走使用 X；不要依赖松键即时停止。终端输入超过 0.6 秒未更新会归零，速度加速过程由控制器限制。

| 阶段 | 相对接管时间 |
|---|---|
| 站姿接管与保持 | 0～7 秒 |
| 初始零速度策略 | 7～8 秒 |
| 键盘输入窗口 | 8～20 秒，累计运动最多 4 秒 |
| 归零观察 | 20～22 秒 |
| 降低与稳定 | 22～29 秒 |
| 阻尼收尾 | 随后 1 秒 |

预算耗尽或输入窗口结束都会强制归零，之后仍执行降低。测试 X 主动停走时，应在预算耗尽前按 X。流程正常结束后现场确认机器人已降低并停住；下一轮需重新恢复站立。

Q/E 用于行进中转向，始终包含向前速度；此入口不提供原地转向。

## 停止与原厂模式恢复

### 第二终端停止

主动真机流程运行期间，在另一终端请求阻尼：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

X/空格是运动模式的速度归零，Z/Ctrl+C 和第二终端请求是阻尼停止。阻尼会使机身降低，两种操作的效果不同。

### 恢复原厂模式

流程退出后，原厂模式不会自动恢复。确认机器人已卧倒、静止且周围空旷，再执行：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --restore-normal --confirm-prone --allow-factory-stand
```

**恢复 `normal` 可能立即让机器人站起。** 此命令检查新鲜且 CRC 有效的低姿态状态、停止接口已退出及当前模式，然后请求恢复原厂控制。只有在实际确认卧倒时才使用 `--confirm-prone`。

原厂模式恢复后，可由 App 操作。每轮再次接管前重新确认站稳；不要在旧控制进程仍运行时恢复模式。

## 策略接口与配置

部署 Actor 接收 45 维观测，顺序与缩放如下：

| 观测 | 维度 | 缩放 |
|---|---|---|
| 机身角速度 | 3 | 0.25 |
| 投影重力 | 3 | 1.0 |
| 前进、侧向、偏航指令 | 3 | 2.0 / 2.0 / 0.25 |
| 关节位置相对默认角度 | 12 | 1.0 |
| 关节速度 | 12 | 0.05 |
| 上一策略动作 | 12 | 1.0 |

Actor 不使用基座线速度。60 维 Critic 在训练时额外使用仿真线速度与关节力矩，不进入部署模型。策略输出 12 个关节位置动作，每 20ms 更新一次：

```text
关节参考目标 = 默认关节角 + 0.25 × 策略动作
```

策略关节顺序为 FL/FR/RL/RR，SDK 顺序为 FR/FL/RR/RL；每条腿依次为 hip、thigh、calf。观测构造与关节映射由[共用策略接口](../common/go2_policy.py)维护。导出的策略与参数必须配套；修改观测顺序、缩放、关节映射、默认角度、动作缩放、增益或周期后，应重新完成对应仿真及部署检查。

真机配置位于 [configs/go2.yaml](configs/go2.yaml)。策略增益为 Kp20/Kd0.5，受限运动另使用 14Nm PD 输出限幅和 16Nm 反馈停止阈值。这些参数用于受限运行流程，不是原厂推荐值。降低后的末态跟踪门槛为：前进及行进转向 0.15rad；本轮执行过后退或侧移时为 0.25rad。

参考关节目标先按模型范围裁剪；原始动作历史保留，实际关节范围、动作变化、姿态、通信和力矩继续独立检查。更换模型或修改配置后，旧基线和前置日志不能直接沿用。

## 运行日志与复核

运行记录保存在本机 `logs/go2_real/`，主动验收记录在 `logs/go2_real/commission/`。日志包含模型与配置摘要、输入事件、策略目标、关节反馈及停止原因；足端字段为未校准原始整数。

足端原始字段未提供单位换算、足序映射和校准，不能直接作为牛顿或精确接触力，也不能依赖 `foot_force_est` 判断承重。

```bash
# 分析本轮流程
python -m deploy.deploy_real.go2_commission --analyze TRIAL_JSONL

# 回放策略输出，不连接机器人
python -m deploy.deploy_real.go2_policy_replay TRIAL_JSONL --output replay.json

# 只读采集足端原始字段与IMU
python -m deploy.deploy_real.go2_contact_diagnostics --network eno1
```

日志分析确认软件流程；机身实际运动和停止需由现场观察或独立位姿测量确认。出现通信超时、姿态异常、撞限位或异常声响时停止测试，检查日志后再处理，不通过放宽阈值掩盖故障。

## 仿真与离线诊断

DDS 仿真使用回环网卡和非零 domain，与真机 domain 0 分离。需先启动兼容的 MuJoCo / DDS 桥接：

```bash
python deploy/deploy_real/deploy_real_go2.py lo \
  --domain-id 1 --simulation-auto --duration 12 --command 0.3 0 0
```

`--simulation-auto` 不允许用于实体网卡。无需 DDS 的受限键盘流程检查：

```bash
python -m deploy.deploy_mujoco.validate_go2_keyboard --unified
```

此检查覆盖六个单方向及跨轴、前后和左右转向切换，需要匹配的本机前置记录，可用 `--readiness ZERO_JSONL` 指定；省略 `--unified` 则只检查后退/侧移模式。报告分别评价流程、指令检查和运动效果；默认运动窗口实际/指令距离比门槛为 0.5，仅作为电脑端工程判据。

仿真报告分别记录指令、实际运动和停止结果。运动效果需结合距离、速度和姿态评价；仿真结果与真机现场检查分别记录。

专项调试还提供 `go2_commission --joint-check`、`--response-test` 与 `--damping-test`。它们用于受限关节响应或停止链路检查，须按各自的姿态、幅度和时长约束执行，不能作为直接启动行走的替代步骤。
