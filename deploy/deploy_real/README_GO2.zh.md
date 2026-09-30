# Go2真机部署

本入口复用Isaac Gym导出的45维Actor、`deploy.yaml`和公共策略接口。代码虽已包含安全检查，实体机器人仍可能造成设备损坏或人身伤害；必须按下列阶段逐级验证，禁止直接落地启动策略。

本程序的键盘速度输入目前仅用于DDS仿真与只读影子检查。独立停机已通过仿真，尚未在实体Go2上验收；真机网卡上仅允许`--read-only`。直接运行`python deploy/deploy_real/deploy_real_go2.py eno1`会在初始化DDS前报错。`--duration`用于自动仿真、受限电机测试和只读影子检查。

宇树手机App可以操作原厂运动功能；只读检查期间可保留原厂控制。本程序尚未读取App虚拟摇杆作为策略速度指令。底层电机控制通常需要退出原厂运动服务，因此App的停止按钮不能未经验证就作为底层策略的停机方式。

## 控制状态

- `Passive`：程序启动状态；单独运行`--read-only`时保持此状态，只记录数据。
- `FixStand`：运行`--shadow-policy`或`--simulation-auto`后自动进入。前者只计算5秒插值目标；后者在仿真中下发目标。
- `Policy`：上述5秒插值完成且安全检查通过后，程序自动进入策略计算；影子模式不下发，DDS仿真模式下发。
- `MotorTest`：仅使用`--motor-test`进入；从当前关节角开始做单关节小幅往返，不运行策略。
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

## 独立停止与受限电机测试

主动DDS仿真现在由独立看门狗进程作为唯一LowCmd发布者。主程序只读取状态、计算目标并发送给看门狗。看门狗独立检查状态、目标角、增益和估算力矩；控制命令超过100毫秒未更新时，即使主程序卡住或被`SIGKILL`结束，也会锁存停止、将Kp归零并持续发送约1秒阻尼。停止后必须重新启动程序才能再次主动控制。

在另一个终端停止domain 1的仿真控制：

```bash
conda activate unitree-rl
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 1
```

命令收到看门狗确认后才输出成功；确认表示阻尼指令已发送，不代表机器人保持站立。该路径仍依赖同一台电脑、看门狗进程和DDS通信，不能覆盖电脑断电、看门狗被强制结束或网络完全中断，也没有验证实体Go2的响应。

已有Go2 DDS仿真运行在`lo`、domain 1时，可以运行受限测试：

```bash
python deploy/deploy_real/deploy_real_go2.py lo --domain-id 1 \
  --motor-test --test-joint FL_hip_joint --test-amplitude 0.01 --duration 2
```

这条命令本身不启动MuJoCo。它以当前关节角为起点，只改变所选关节的目标；其余关节以低增益保持采样时的姿态。默认目标偏移0.01弧度（约0.57度），可设置的最大绝对偏移为0.02弧度，时长1至3秒；使用平滑往返曲线，Kp=5、Kd=0.5。主动测试期间，实际关节位移超过0.04弧度、关节速度超过2弧度/秒、估算PD力矩超过1牛米或出现其他安全异常时进入阻尼。目标位移限制不能保证实际位移或跌倒风险；这些参数尚未通过真机验收。

可直接运行自动验收，无需机器人或额外仿真程序：

```bash
GO2_DDS_TESTS=1 python -m unittest discover -s tests -p test_go2_watchdog.py -v
```

测试使用本项目Go2 MJCF资产和官方SDK消息，自动启动无窗口的MuJoCo/DDS测试桥接，覆盖单关节往返、地面偏移超限、第二终端停止、主程序冻结、主程序强制结束和LowState断流。单关节方向检查采用固定机身、零重力条件；地面场景单独保留重力，验证低增益导致偏移超限时会停止。这些测试不证明无支撑真机测试可行。

## 真机分阶段验收

1. 完成上面的只读与策略影子检查，逐项核对实体IMU、关节顺序、App虚拟摇杆的数据来源和状态更新频率。
2. 受限电机测试入口和独立停机已完成仿真验证；目前均限定在`lo`和非零DDS domain。真机主动控制仍被启动前检查阻止。
3. 首次电机控制前，还需核对机器人高层运动服务的关闭方式、可在实体机上使用的独立停机方式和退出后的状态，并确定适合实际摆放条件的受限测试方案。地面仿真中低增益无法保证承重站立，不能照搬为无支撑真机测试。
4. 受限电机控制与急停在真机通过后，再逐步测试站立、停止、微速移动及策略；任一步异常都停止并检查日志。

DDS仿真停机时由看门狗持续发送约1秒阻尼命令；真机只读模式始终不发送电机命令。主程序的计算记录保存在 `logs/go2_real/*.jsonl`；实际发布命令的记录位于 `logs/go2_real/watchdog/*.jsonl`，包含IMU、关节状态、目标角、增益、控制状态和停止原因。

## 重要限制

- 当前代码尚未经过实体Go2验收；完成DDS仿真不等于已完成真机部署。
- 不得修改45维观测顺序、策略↔SDK关节映射或20 ms策略周期。
- 不得同时运行其他LowCmd控制程序。
- 出现重复超时、姿态异常、关节撞限位或异常声响时立即急停并检查日志，不得通过放宽阈值掩盖问题。
