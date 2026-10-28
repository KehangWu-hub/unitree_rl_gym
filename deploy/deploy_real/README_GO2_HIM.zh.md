# Go2 HIM训练与部署

[项目首页](../../README.md) · [环境安装](../../doc/setup_zh.md)

本指南介绍HIM训练、导出、MuJoCo运行和真机键盘控制。所有命令从项目根目录执行：

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 训练

使用混合地形训练配置：

```bash
python legged_gym/scripts/train_him_from_scratch.py \
  --profile candidate-rough --run-dir logs/him/my_run \
  --headless --num_envs 1024 --max_iterations 3000 --seed 42
```

| 参数 | 作用 |
|---|---|
| `--profile candidate-rough` | 混合地形与地形课程 |
| `--profile candidate-flat` | 对应的平地配置 |
| `--num_envs` | 并行环境数，按显存选择 |
| `--max_iterations` | 本次训练轮数 |
| `--run-dir` | 独立运行目录，每次新训练使用新目录 |
| `--background` | 后台运行，输出写入运行目录 |

运行目录保存配置、源码快照、TensorBoard记录和断点；训练结束后导出：

```text
logs/him/my_run/
├── env.yaml
├── train.yaml
├── progress.json
├── model_3000.pt
└── exported/
    ├── policies/policy.pt
    └── params/deploy.yaml
```

从已完成的运行继续训练到总计5000轮：

```bash
python legged_gym/scripts/continue_him_training.py \
  --source-run logs/him/my_run --run-dir logs/him/my_run_extended \
  --target-iteration 5000 --background
```

原PPO及基础HIM任务也可通过通用入口运行：`python legged_gym/scripts/train.py --task go2_him --headless`。

## MuJoCo运行

先用仓库附带的配套模型运行：

```bash
python -m deploy.deploy_mujoco.deploy_go2 \
  --config deploy/deploy_mujoco/configs/go2_him_fresh.yaml \
  --command 0.3 0 0
```

使用自己的训练结果时，成套指定模型和接口参数：

```bash
python -m deploy.deploy_mujoco.deploy_go2 \
  --config deploy/deploy_mujoco/configs/go2_him_fresh.yaml \
  --policy logs/him/my_run/exported/policies/policy.pt \
  --deploy-config logs/him/my_run/exported/params/deploy.yaml \
  --command 0.3 0 0
```

`--command VX VY WZ`分别是前进、侧移和偏航指令。可添加 `--headless --no-realtime --duration 10` 输出无窗口仿真统计。

## 真机键盘控制

机器人通过网线连接电脑，从原厂站稳姿态接管；现场保持空旷，并准备停止终端。下文网卡以 `eno1` 为例，替换为实际有线网卡名。

选择仓库中的一组HIM配置与模型清单：

```bash
HIM_CONFIG=deploy/deploy_real/configs/go2_him_fresh.yaml
HIM_BASELINE=deploy/deploy_real/configs/go2_him_fresh_baseline.json
```

检查配置与模型，不初始化DDS：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --config "$HIM_CONFIG" --baseline "$HIM_BASELINE" --check
```

机器人连接后，可只读检查状态：

```bash
python -m deploy.deploy_real.deploy_real_go2 eno1 \
  --config "$HIM_CONFIG" --read-only --duration 5
```

启动键盘控制：

```bash
python -m deploy.deploy_real.go2_keyboard \
  --network eno1 --config "$HIM_CONFIG" --baseline "$HIM_BASELINE" --run
```

根据终端提示输入 `START`，待按键窗口开启后控制。切换模型时，配置和清单必须成套选择。

| 按键 | 指令或操作 |
|---|---|
| W / S | 前进 / 后退，±0.30m/s |
| A / D | 左移 / 右移，±0.30m/s |
| Q / E | 前进0.30m/s，同时转向±0.30rad/s |
| X / 空格 | 速度归零，策略继续运行 |
| C | 受控降低并退出 |
| Z / Ctrl+C | 立即切换阻尼并退出 |

最后一个方向键生效；主动停步使用X或空格。独立倾角保护触发后锁定阻尼，受控退出过程中仍可按Z停止。

第二终端停止命令：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

恢复原厂模式及网络说明见[PPO部署指南](README_GO2.zh.md#停止与恢复原厂模式)。

## 模型输入输出

TorchScript包含历史编码器与Actor，输入为 `[batch, 270]`：六帧45维观测，最新帧在前。每帧按以下顺序拼接：

| 观测 | 维度 |
|---|---:|
| 前进、侧向、偏航指令 | 3 |
| 机身角速度 | 3 |
| 投影重力 | 3 |
| 关节位置相对默认角度 | 12 |
| 关节速度 | 12 |
| 上一策略动作 | 12 |

编码器输出3维估计速度和16维隐变量，与当前45维观测共同构成64维Actor输入。模型输出 `[batch, 12]`，关节参考目标为默认关节角加 `动作缩放 × 策略动作`，缩放及关节顺序以配套 `deploy.yaml` 为准。

训练时Critic使用238维观测：基础观测45＋真实线速度3＋外部扰动力3＋地形高度187；Critic不参与部署推理。

## 替换部署权重

HIM部署包包含 `policy.pt`、`deploy.yaml` 和 `provenance.json`。真机使用的完整YAML配置还包括网卡、控制及停止参数；模型包的 `deploy.yaml` 不能直接替代它。

更换权重时，同步更新模型包、真机YAML中的接口字段及匹配的baseline摘要。真机配置中的部署状态字段控制模型是否允许启动。
