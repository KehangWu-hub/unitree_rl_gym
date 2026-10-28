# Go2 PPO真机控制

[项目首页](../../README.md) · [环境安装](../../doc/setup_zh.md) · [HIM训练与部署](README_GO2_HIM.zh.md)

本指南对应 `configs/go2.yaml` 和PPO部署模型。HIM使用独立的配置与模型清单，见[HIM指南](README_GO2_HIM.zh.md)。所有命令从项目根目录执行。

## PPO训练与导出

训练需要安装Isaac Gym及rsl-rl：

```bash
python legged_gym/scripts/train.py \
  --task go2 --headless --num_envs 1024 --max_iterations 1500

python legged_gym/scripts/play.py \
  --task go2 --headless --load_run RUN_DIRECTORY --checkpoint 1500 --export_only
```

将 `RUN_DIRECTORY` 替换为本次训练的目录名。程序打印导出的模型和 `deploy.yaml` 路径，部署时使用这组配套文件。

## 连接与准备

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

通过网线连接机器人，使用同一子网内的电脑地址，DDS domain为0。下文使用 `eno1`，请替换为实际有线网卡。机器人应处于平坦空地，原厂站稳，现场有人观察且可停止。

检查本地模型与配置：

```bash
python -m deploy.deploy_real.deploy_real_go2 --check
```

只读查看机器人状态：

```bash
python -m deploy.deploy_real.deploy_real_go2 eno1 --read-only --duration 5
```

## 首次配置

PPO键盘入口需要匹配当前模型的站姿和零速记录。记录由本机生成，程序输出实际JSONL路径。

站姿接管：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --stand-check --start-standing --confirm-standing \
  --release-mode --allow-body-movement
```

流程结束后，按下文恢复原厂模式并重新站稳。将 `STAND_JSONL` 替换为上一步生成的日志路径，运行零速策略：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --mode zero --readiness STAND_JSONL --run
```

后续运动使用该步骤输出的 `ZERO_JSONL`。这些记录用于匹配模型与配置；更换模型后重新生成。

## 键盘控制

恢复原厂站稳后，先检查配置与记录，再启动：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --readiness ZERO_JSONL --check

python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --readiness ZERO_JSONL --run
```

根据终端提示输入 `START`，待按键窗口开启后控制。已有匹配本机记录时可省略 `--readiness`。

| 按键 | 指令或操作 |
|---|---|
| W / S | 前进 / 后退，±0.50m/s |
| A / D | 左移 / 右移，±0.40m/s |
| Q / E | 前进0.50m/s，同时转向±0.50rad/s |
| X / 空格 | 速度归零，策略继续运行 |
| C | 受控降低并退出 |
| Z / Ctrl+C | 阻尼停止并退出 |

默认持续运行至主动退出。Q/E包含前进速度；终端超过0.6秒未收到方向输入会归零，主动停步使用X或空格。接管期间由本项目控制器发送命令，手机App不作为停止入口。

## 停止与恢复原厂模式

第二终端可随时请求阻尼停止：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

C正常降低后退出；Z、Ctrl+C和第二终端停止立即撤除位置支撑并进入阻尼。控制进程退出、机器人已趴稳后，恢复原厂模式：

```bash
python -m deploy.deploy_real.go2_commission eno1 \
  --restore-normal --confirm-prone --allow-factory-stand
```

恢复操作可能直接使机器人站起；完成后可重新使用App控制。

## 模型与配置

| 文件 | 用途 |
|---|---|
| `deploy/pre_train/go2/motion.pt` | PPO部署模型 |
| `deploy/deploy_real/configs/go2.yaml` | 真机模型路径、观测与控制参数 |
| `deploy/deploy_real/configs/go2_verified_baseline.json` | 模型、配置和记录摘要 |
| `logs/go2_real/` | 本地运行日志 |

PPO输入为45维观测：角速度3、投影重力3、指令3、关节位置12、关节速度12、上一动作12。输出12维关节位置动作，参考目标为默认关节角加 `0.25 × 动作`。

策略关节顺序为FL/FR/RL/RR，SDK顺序为FR/FL/RR/RL，映射由共用策略接口处理。策略周期20ms，PD增益20/0.5；运行约束由所选配置与速度模式决定。

## 可选工具

| 入口 | 用途 |
|---|---|
| `go2_commission --analyze TRIAL_JSONL` | 分析本机控制日志 |
| `go2_velocity_diagnostics --analyze TRIAL_JSONL` | 查看速度观测 |
| `go2_policy_replay TRIAL_JSONL --output replay.json` | 离线回放策略输入输出 |
| `go2_keyboard --velocity-source kinematic --velocity-feedback shadow` | 记录运动学速度估计 |

以上入口均位于 `deploy.deploy_real`，通过 `python -m deploy.deploy_real.<入口>` 使用。参数说明可通过对应入口的 `--help` 查看。
