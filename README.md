<div align="center">
  <h1 align="center">Unitree Go2 RL Gym</h1>
</div>

<p align="center">
  <strong>用于 Unitree Go2 的强化学习训练、仿真与真机部署。</strong>
</p>

> [!IMPORTANT]
> 本项目基于 Unitree 官方的
> [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
> 进行修改，用于 Unitree Go2 的强化学习训练、仿真验证和真机部署。
> **本项目不是 Unitree 官方项目，与 Unitree Robotics 无隶属或背书关系。**

## 项目说明

本仓库在上游 `unitree_rl_gym` 的基础上进行了针对 Unitree Go2 的适配，
覆盖策略训练、效果预览、Sim2Sim 验证以及真机部署流程。除非另有说明，
从上游项目保留的代码和文档仍遵循原有版权声明与许可证条款。

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

</div>

---

## 📦 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️  参数说明
- `--task`: 必选参数，值可选(go2, g1, h1, h1_2)
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py --task=xxx
```

**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`
  
### Play 效果

| Go2 | G1 | H1 | H1_2 |
|--- | --- | --- | --- |
| [![go2](https://oss-global-cdn.unitree.com/static/ba006789e0af4fe3867255f507032cd7.GIF)](https://oss-global-cdn.unitree.com/static/d2e8da875473457c8d5d69c3de58b24d.mp4) | [![g1](https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF)](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [![h1](https://oss-global-cdn.unitree.com/static/fa04e73966934efa9838e9c389f48fa2.GIF)](https://oss-global-cdn.unitree.com/static/522128f4640c4f348296d2761a33bf98.mp4) |[![h1_2](https://oss-global-cdn.unitree.com/static/83ed59ca0dab4a51906aff1f93428650.GIF)](https://oss-global-cdn.unitree.com/static/15fa46984f2343cb83342fd39f5ab7b2.mp4)|

---

### 3. Sim2Sim (Mujoco)

支持在 Mujoco 仿真器中运行 Sim2Sim：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py {config_name}
```

#### 参数说明
- `config_name`: 配置文件，默认查询路径为 `deploy/deploy_mujoco/configs/`

#### 示例：运行 G1

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

#### ➡️  替换网络模型

默认模型位于 `deploy/pre_train/{robot}/motion.pt`；自己训练模型保存于`logs/g1/exported/policies/policy_lstm_1.pt`，只需替换 yaml 配置文件中 `policy_path`。

#### 运行效果

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![mujoco_g1](https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF)](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4)  |  [![mujoco_h1](https://oss-global-cdn.unitree.com/static/7ab4e8392e794e01b975efa205ef491e.GIF)](https://oss-global-cdn.unitree.com/static/8934052becd84d08bc8c18c95849cf32.mp4)  |  [![mujoco_h1_2](https://oss-global-cdn.unitree.com/static/2905e2fe9b3340159d749d5e0bc95cc4.GIF)](https://oss-global-cdn.unitree.com/static/ee7ee85bd6d249989a905c55c7a9d305.mp4) |


---

### 4. Sim2Real (实物部署)

实现实物部署前，确保机器人进入调试模式。详细步骤请参考 [实物部署指南](deploy/deploy_real/README.zh.md)：

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```

#### 参数说明
- `net_interface`: 连接机器人网卡名称，如 `enp3s0`
- `config_name`: 配置文件，存在于 `deploy/deploy_real/configs/`，如 `g1.yaml`，`h1.yaml`，`h1_2.yaml`

#### 运行效果

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

---

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
- [unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): 实物部署硬件通信接口。


---

## 🔖  许可证

本项目根据 [BSD 3-Clause License](./LICENSE) 授权：
1. 必须保留原始版权声明。
2. 禁止以项目名或组织名作举。
3. 声明所有修改内容。

详情请阅读完整 [LICENSE 文件](./LICENSE)。
