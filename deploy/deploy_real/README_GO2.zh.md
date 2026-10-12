# Go2 真机部署指南

本文介绍 Go2 策略部署的有线连接、前置检查、键盘控制和停止操作。所有命令均从项目根目录运行，使用项目的 `unitree-rl` Conda 环境，并确保已安装 `unitree_sdk2py`。

[项目首页](../../README.md) · [环境安装](../../doc/setup_zh.md)

当前真机发布基线为 [v1.0](../../doc/releases/v1.0.md)，现场操作者已确认运动效果验收通过；冻结文件摘要见 [版本清单](configs/go2_v1.0_manifest.json)。

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

默认 `all` 模式在同一轮接受 Q/W/E/A/S/D，无需重新启动来切换方向。W/S 前进或后退，A/D 左右侧移，默认平移指令上限为 0.30m/s。Q/E 向前移动并叠加左右转向，上限为 0.30m/s 和 ±0.30rad/s。

`--walk-speed 0.5` 将 W 纯前进指令设为 0.50m/s，仅适用于 `all` 模式。S/A/D 和 Q/E 保持各自的 0.30 指令上限。所有运动入口默认采用 `limit` 动作处理；显式 `--action-step-mode limit` 仍可使用。`reject` 仅保留对照用途，不允许与 0.50m/s 档组合。

动作限幅先约束合法策略输出的步长，再计算满足角度与PD力矩范围的参考目标。按当前20ms周期、0.25动作缩放，每步动作最多0.64，对应参考角变化最多0.16rad；这是参考轨迹整形参数，不是实际关节速度上限。限幅后的动作进入下一周期last_action，原始动作另行记录。所有运动档的实际速度检查按关节取URDF与配置的交集，当前髋/大腿25rad/s、小腿20.07rad/s，不随W速度或方向改变。前进指令以1.0m/s²加速；指令不保证实际移动速度。

例如先离线检查：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --walk-speed 0.5 --check
```

每次只执行最后收到的方向键，不组合多个按键。方向改变时先归零一个策略周期，再向新方向加速。X/空格将所有速度分量归零；从转向切回平移时不会保留转向指令。

统一模式默认持续运行：接管后第8秒开放按键，无累计运动预算、输入窗口末尾或自动降低时间。由用户按C正常退出，或按Z/Ctrl+C及第二终端紧急终止。X/空格只将速度归零，仍可继续输入其他方向。

在已激活unitree-rl环境的项目目录终端中，可使用一行命令：

```bash
python -m deploy.deploy_real.go2_keyboard --walk-speed 0.5 --run
```

如需短时自动结束的检查，可显式选择`--motion-seconds 4`或`10`。短时模式中，切换方向或按X不会重置累计预算；预算耗尽后指令归零，输入窗口结束后自动降低。该选项仅作用于all模式。

### 单方向模式

可用以下短时检查模式限制允许输入的方向键，前置记录与统一模式相同；这些模式沿用4秒预算和自动降低：

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
| C | 持续模式正常退出：归零2秒，受控降低5秒、稳定2秒，然后退出 |
| Z / Ctrl+C | 紧急终止，立即进入阻尼 |

输入 `START` 后暂不操作 App。看到方向键窗口提示再按键，主动停走使用 X；不要依赖松键即时停止。终端输入超过 0.6 秒未更新会归零，速度加速过程由控制器限制。

以下时序仅用于显式短时检查；默认持续模式在第8秒开放按键后一直等待用户退出。

| 阶段 | 可选 4 秒档 | 可选 10 秒档 |
|---|---|---|
| 站姿接管与保持 | 0～7 秒 | 0～7 秒 |
| 初始零速度策略 | 7～8 秒 | 7～8 秒 |
| 键盘输入窗口 | 8～20 秒 | 8～26 秒 |
| 归零观察 | 20～22 秒 | 26～28 秒 |
| 降低与稳定 | 22～29 秒 | 28～35 秒 |
| 阻尼收尾 | 随后 1 秒 | 随后 1 秒 |

短时检查的预算耗尽或输入窗口结束都会强制归零，之后仍执行降低。测试 X 主动停走时，应在预算耗尽前按 X。流程正常结束后现场确认机器人已降低并停住；下一轮需重新恢复站立。

Q/E 用于行进中转向，始终包含向前速度；此入口不提供原地转向。

## 停止与原厂模式恢复

### 第二终端停止

主动真机流程运行期间，在另一终端请求阻尼：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

X/空格是运动模式的速度归零，C 是受控降低后退出，Z/Ctrl+C 和第二终端请求是紧急阻尼停止。阻尼会使机身降低，两种操作的效果不同。

### 恢复原厂模式

流程退出后，原厂模式不会自动恢复。确认机器人已卧倒、静止且周围空旷，再执行：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --restore-normal --confirm-prone --allow-factory-stand
```

**恢复 `normal` 可能立即让机器人站起。** 此命令最多等待 15 秒，要求新鲜且 CRC 有效的状态连续 1 秒满足静止低姿态条件，并检查停止接口已退出及当前模式，然后请求恢复原厂控制。超时提示会列出未满足的条件。只有在实际确认卧倒时才使用 `--confirm-prone`。

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

参考关节目标先按模型范围裁剪，再投影到角度与PD力矩允许区间的交集；实际关节范围、姿态、通信和反馈力矩独立检查。limit模式的last_action使用步长处理后的动作，reject模式沿用原始动作；两者均不以PD限幅后的目标反算动作历史。更换模型或修改配置后，旧基线和前置日志不能直接沿用。

运行中的限制按阶段区分，不能把站姿检查门槛当作步态速度上限：

| 限制/触发条件 | 来源及适用阶段 | 处理 |
|---|---|---|
| 关节参考角范围 | 配置与Go2 URDF一致 | 裁剪参考角；实际角度超出范围及0.02rad测量余量时停止 |
| 实际关节速度 | 运动策略：逐关节min(URDF, 配置25rad/s) | 超限停止并记录关节名、实测值和上限 |
| 动作步长0.64 | 软件参考轨迹约束，所有运动策略默认limit | 限幅后继续，不因合法输出跳变直接阻尼 |
| 动作幅值4.0、NaN/Inf | 部署输出有效性检查 | 仍停止，不用限幅隐藏异常输出 |
| PD 14Nm、反馈16Nm | 项目受限运行配置，并非硬件额定范围 | PD先限幅；角度/PD无可行交集或反馈超限时停止 |
| 内层PD 0.8×模型力矩 | 配置的最终命令检查 | 14Nm限幅后仍不满足时停止 |
| 倾角0.35rad | 接管、策略、降低流程的稳定性门槛 | 超限停止；未当作机器人机械范围撤除 |
| 关节速度2rad/s | 无运动指令的零速验收、站姿接管和降低阶段 | 超限停止；不用于运动策略阶段 |
| 跟踪0.25rad；降低末态0.15/0.25rad | 站姿/降低流程完成条件 | 不满足则停止，不伪报正常完成 |
| 状态/父进程心跳0.1秒、发布循环0.05秒 | 通信和执行有效性 | 停止；不再依赖失效的策略维持站姿 |
| 按键超时、X、可选短时预算耗尽 | 输入及可选时间约束 | 速度指令归零，策略继续，随后按计划降低 |

可直接从一个方向键切换到另一个方向键，不要求中间按X。换向会先将指令归零一个20ms周期，再向新方向加速；这不保证机身已物理停稳。X用于主动停走。

显式短时模式结束前执行计划内降低；默认持续模式按C正常退出：接管完成后的策略阶段接受C，速度归零2秒，再沿用5秒降低及2秒稳定，最后一秒阻尼收尾；C重复输入不会重置过程，期间方向键不再接受。Z/Ctrl+C或独立停止可随时紧急终止。故障和紧急终止立即进入阻尼。阻尼取消主动位置支撑，可能下沉，不等于站立刹停。故障停止保持锁存，不自动恢复站立。日志中的`policy_joint_velocity_limits_sdk_rad_s`保存逐关节上限，`policy_limits_version`标识限制实现版本；历史日志不因此扩大已验证范围。


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


持续入口的电脑端仿真使用同一个仓库脚本。例如验证W、X与C受控降低：

```bash
python deploy/deploy_mujoco/validate_go2_keyboard.py --unified --continuous --keys w --walk-speed 0.5 --exit-time 20
```

`--exit-key z`改为模拟紧急终止；`--exit-time`只决定电脑端仿真何时触发退出。仿真没有图形窗口，输出流程、运动效果及日志检查，不代表真机通过。
