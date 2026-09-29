# Go2真机部署

本入口复用Isaac Gym导出的45维Actor、`deploy.yaml`和公共策略接口。代码虽已包含安全检查，实体机器人仍可能造成设备损坏或人身伤害；必须按下列阶段逐级验证，禁止直接落地启动策略。

本程序的键盘速度输入目前仅用于DDS仿真与只读影子检查。由于独立停机方式尚未验收，真机网卡上仅允许`--read-only`；直接运行`python deploy/deploy_real/deploy_real_go2.py eno1`会在初始化DDS前报错。`--duration`只对自动仿真和只读影子检查生效。

宇树手机App可以操作原厂运动功能；只读检查期间可保留原厂控制。本程序尚未读取App虚拟摇杆作为策略速度指令。底层电机控制通常需要退出原厂运动服务，因此App的停止按钮不能未经验证就作为底层策略的停机方式。

## 控制状态

- `Passive`：程序启动状态；单独运行`--read-only`时保持此状态，只记录数据。
- `FixStand`：运行`--shadow-policy`或`--simulation-auto`后自动进入。前者只计算5秒插值目标；后者在仿真中下发目标。
- `Policy`：上述5秒插值完成且安全检查通过后，程序自动进入策略计算；影子模式不下发，DDS仿真模式下发。
- `Damping`：发生通信、姿态、关节、动作或力矩异常，或退出仿真程序时进入。只读模式下仅在程序内部记录状态，不发送电机命令。

键盘`W/S`、`A/D`、`Q/E`分别给出前后、侧向和转向速度；当前上限分别为0.5米/秒、0.3米/秒和0.5弧度/秒。

## 只检查模型，不启动DDS

```bash
conda activate unitree-rl
python deploy/deploy_real/deploy_real_go2.py --check
```

该命令必须先通过。它不会连接仿真器或机器人，也不会创建LowCmd发布器。

## 真机只读与策略影子检查

在机器人使用原有控制方式保持稳定时，连接有线网卡 `eno1`，先运行：

```bash
conda activate unitree-rl
python deploy/deploy_real/deploy_real_go2.py eno1 --read-only
```

该模式不创建LowCmd发布器。保持机器人静止，观察约10秒后按终端`Ctrl+C`结束；不要为测试本程序而触发原厂运动操作。然后运行：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 --read-only --shadow-policy --duration 12
```

该命令自动在程序内部经历5秒虚拟`FixStand`，随后以零速度指令计算约7秒策略目标，不向机器人发送电机命令。检查新生成的`logs/go2_real/*.jsonl`中出现`policy`记录，且`reason`为空、目标角均为有限值。若未进入`policy`或有安全故障，先检查输入和日志。

只读阶段不要关闭机器人原有的运动服务；它仍负责保持机器人姿态。

## 键盘输入预演（不连接真机电机）

在官方DDS/MuJoCo仿真启动后，可在前台交互式终端运行：

```bash
python deploy/deploy_real/deploy_real_go2.py lo \
  --domain-id 1 --simulation-auto --keyboard --duration 20
```

程序先执行5秒虚拟站立，再进入策略。`W/S`前后、`A/D`左右、`Q/E`转向；`X`或空格把速度目标归零，策略继续运行并保持站立。每次方向按键只维持0.25秒，未持续收到按键时自动归零；终端无法可靠报告松键事件，因此按键重复间隔会影响连续运动。`Z`或`Ctrl+C`退出并发送约1秒阻尼命令；仿真中的机器人随后会降低机身，不能把它当作保持站立的停车方式。

2026-09-29的动态DDS/MuJoCo测试已验证两次键盘前进、`X`归零和空格归零：策略运行359帧，软件安全故障为零；停止行走期间机身高度约0.32米。`Z`退出后仿真机器人降低机身。该结果只适用于所用MuJoCo模型和测试指令，不代表真机停机已验收。

下次真机开机时，可在只读影子模式下检查键盘输入是否进入策略日志：

```bash
python deploy/deploy_real/deploy_real_go2.py eno1 \
  --read-only --shadow-policy --keyboard --duration 20
```

这个模式不会创建LowCmd发布器。键盘的`X`、空格和超时归零只是速度目标变化，不能充当真实电机急停。真机主动控制的启动检查仍会阻止`--keyboard`绕过只读限制。

## DDS仿真验证

先在独立终端启动官方 `unitree_mujoco`，机器人选择Go2，DDS domain使用1，网卡使用本地回环 `lo`。随后运行：

```bash
conda activate unitree-rl
python deploy/deploy_real/deploy_real_go2.py lo \
  --domain-id 1 --simulation-auto --duration 12 \
  --command 0.3 0.0 0.0
```

必须在DDS仿真中检查状态机、键盘速度与归零、关节映射、CRC、控制周期、停止与异常阻尼，不允许跳过此阶段连接真机。

## 真机分阶段验收

1. 完成上面的只读与策略影子检查，逐项核对实体IMU、关节顺序、App虚拟摇杆的数据来源和状态更新频率。
2. 当前程序没有独立的限幅短时电机测试入口，真机主动控制已被启动前检查阻止。键盘入口不会解除这一保护。
3. 首次电机控制前，应单独实现并在仿真中验证受限测试入口，同时确认机器人高层运动服务的关闭方式、独立停机方式和退出后的安全状态。悬空支撑可以降低首次测试风险；无支撑地面测试仍有跌倒和损坏风险。
4. 受限电机控制与急停在真机通过后，再逐步测试站立、停止、微速移动及策略；任一步异常都停止并检查日志。

DDS仿真主动控制程序退出时会持续发送约1秒阻尼命令；真机只读模式始终不发送电机命令。运行数据保存在 `logs/go2_real/*.jsonl`，包含IMU、关节状态、速度指令、目标角、增益、状态机状态和安全降级原因。

## 重要限制

- 当前代码尚未经过实体Go2验收；完成DDS仿真不等于已完成真机部署。
- 不得修改45维观测顺序、策略↔SDK关节映射或20 ms策略周期。
- 不得同时运行其他LowCmd控制程序。
- 出现重复超时、姿态异常、关节撞限位或异常声响时立即急停并检查日志，不得通过放宽阈值掩盖问题。
