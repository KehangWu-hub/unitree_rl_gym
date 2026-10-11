# Go2 真机部署指南

本文介绍冻结策略的有线连接、前置检查、受限键盘控制和停止操作。所有命令均从项目根目录运行，使用项目的 `unitree-rl` Conda 环境，并确保已安装 `unitree_sdk2py`。

[项目首页](../../README.md) · [环境安装](../../doc/setup_zh.md) · [验证与技术说明](../../doc/go2_validation.md)

## 运行前准备

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

机器人应处于平坦空地，网线保持松弛，有人在旁观察，并准备独立的停止终端。每次主动控制前确认当前姿态；键盘运动从原厂站稳开始，不从卧倒姿态直接启动。

真机使用有线网卡和 DDS domain 0。下文以 `eno1` 为例，请替换成实际网卡。测试进程不可与其他自定义 LowCmd 发布程序同时运行。

控制器接管后会释放原厂运动服务，手机 App 的运动控制可能暂时不可用。主动停止使用键盘或下文的停止终端。软件停止仍依赖电脑、进程和通信，不能覆盖电脑断电或完整物理断链。

## 检查模型和连接

先检查配置与冻结策略，不初始化 DDS：

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

`go2_keyboard` 会核对模型、配置和前置日志。[基线清单](configs/go2_verified_baseline.json) 保存对应摘要，历史 JSONL 日志不随仓库分发。新环境需依次完成站姿接管和零速度策略检查，再将产生的实际日志路径传入后续命令。

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

### 行进中转向

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --readiness ZERO_JSONL --run
```

默认 `walk-turn` 模式接受 Q/E：两者都向前移动，分别叠加左转或右转。前进指令为 0.30m/s，转向指令为 ±0.30rad/s。

### 后退

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode backward --readiness ZERO_JSONL --run
```

S 给出最高 0.30m/s 的后退指令。

### 左右侧移

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode lateral --readiness ZERO_JSONL --run
```

A/D 分别给出最高 0.30m/s 的左移、右移指令。

### 输入与时序

每个运动模式只接收对应方向键；默认模式不接收 W/A/S/D。

| 输入 | 作用 |
|---|---|
| Q / E | 行进中左转 / 右转，限默认模式 |
| S | 后退，限 `backward` 模式 |
| A / D | 左移 / 右移，限 `lateral` 模式 |
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

纯前进 W 属于 `go2_commission` 的历史键盘流程，已有旧参数下的短时记录；当前简短入口不提供 W 模式。相关参数范围见[验证说明](../../doc/go2_validation.md)。

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

## 配置与日志

真机配置位于 [configs/go2.yaml](configs/go2.yaml)。策略增益为 Kp20/Kd0.5，受限运动另使用 14Nm PD 输出限幅和 16Nm 反馈停止阈值。这些是本项目短时测试参数，不是原厂推荐值。

参考关节目标先按模型范围裁剪；原始动作历史保留，实际关节范围、动作变化、姿态、通信和力矩继续独立检查。更换模型或修改配置后，旧基线和前置日志不能直接沿用。

运行记录保存在本机 `logs/go2_real/`，主动验收记录在 `logs/go2_real/commission/`。日志包含模型与配置摘要、输入事件、策略目标、关节反馈及停止原因；足端字段为未校准原始整数。

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
python -m deploy.deploy_mujoco.validate_go2_keyboard
```

此检查需要匹配的本机前置记录，可用 `--readiness ZERO_JSONL` 指定。报告分别评价流程、方向和运动效果；默认运动窗口实际/指令距离比门槛为 0.5，仅作为电脑端工程判据。

专项调试还提供 `go2_commission --joint-check`、`--response-test` 与 `--damping-test`。它们用于受限关节响应或停止链路检查，须按各自的姿态、幅度和时长约束执行，不能作为直接启动行走的替代步骤。
