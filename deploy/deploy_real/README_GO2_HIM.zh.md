# Go2 HIM 仿真与真机验收

当前候选没有完成可靠真机部署。v3在MuJoCo复现停步失稳后仍被送到现场，实机翻车后继续输出位置控制，这是部署与保护设计失误。v3已通过`hardware_deployment_blocked`封停，禁止再次接管；历史失败日志与视频保留。

## 当前运行方式

- 外加14Nm计算PD投影、16Nm反馈停止、动作幅值/步长限幅、目标角裁剪及关节速度/跟踪阈值仍不应用于当前直接目标模式。
- 独立摔倒保护始终检查横滚/俯仰，默认阈值0.35rad（约20度），不受`software_motion_limits`影响。接管、策略、C退出与独立看门狗覆盖，触发后锁定Kp=0、Kd=3阻尼；不能继续执行策略或降低轨迹。仅完成离线验证，不保证所有跌倒都能被阻止。
- 当前直接目标模式不做速度裁剪、组合指令白名单、键盘加速/换向限制；已封停候选始终拒绝真机发布和接管。
- 保留模型本身的预处理与有界输出、20ms推理周期、PD20/0.5、关节映射、启动插值和C受控降低。键盘0.30档是输入指令设置，不作为控制器裁剪上限。
- 保留手动X/C/Z/Ctrl+C、DDS CRC/消息来源核对、有限数值检查和失联/心跳处理。没有修改机器人固件，也不把URDF参数冒充实测固件保护范围。
- 旧配置中的限制仅用于历史复现；当前模型包里的14Nm/0.64字段记录训练时的约束，在`software_motion_limits: false`运行时不应用。

历史4855帧回放和7组MuJoCo报告属于早期v1，不能作为v3通过依据。v3的MuJoCo7组仅6组通过，真机翻车；倾角保护修复后全套离线195项检查中192通过、3项可选DDS跳过，未再次操作机器人。

使用 `unitree-rl` 环境，从项目根目录运行。HIM模型位于 `deploy/pre_train/go2_him_fresh/`，
模型记录及SHA256保存在该目录的 `provenance.json` 中。替换训练权重后，需要重新完成下述仿真与部署验收。
完整模型输入270维历史，内部估计3维速度和16维隐变量，Actor输入64维、输出12维。

## 仿真与离线检查

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m deploy.deploy_real.check_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --output logs/go2_him_fresh/phase3_preflight/summary.json
```

此检查不初始化DDS。它核对模型、历史观测、关节映射、SDK CRC、推理耗时和停止保护，
并验证仿真报告与模型、配置及控制代码的摘要一致。结果在 `logs/go2_him_fresh/phase3_preflight/summary.json`。
更换权重或修改控制代码后，需要重新生成仿真证据。

重跑完整仿真验收：先停止训练，单独完成Isaac Gym评估，再执行MuJoCo测试。

```bash
python legged_gym/scripts/evaluate_him.py --task go2_him_retrain --policy-path deploy/pre_train/go2_him_fresh/policy.pt --output-path logs/go2_him_fresh/phase2_validation/isaac.json --headless --num_envs 64
python legged_gym/scripts/evaluate_him_candidate.py --policy-path deploy/pre_train/go2_him_fresh/policy.pt --command-profile keyboard --output-path logs/go2_him_fresh/phase2_validation/isaac_acceptance.json --headless --num_envs 64
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --output logs/go2_him_fresh/phase2_validation/final_raw
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --protected --keys zero w s a d q e switch --output logs/go2_him_fresh/phase2_validation/final_protected
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --protected --stage zero --output logs/go2_him_fresh/phase2_validation/zero_stage
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --protected --stage forward --output logs/go2_him_fresh/phase2_validation/forward_stage
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --protected --long-trial --output logs/go2_him_fresh/phase2_validation/final_long_keyboard
cp logs/go2_him_fresh/phase2_validation/final_long_keyboard/summary.json logs/go2_him_fresh/phase2_validation/mujoco_acceptance.json
python -m deploy.deploy_mujoco.validate_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --output logs/go2_him_fresh/phase2_validation --qualify
python -m deploy.deploy_real.check_go2_him --config deploy/deploy_real/configs/go2_him_fresh.yaml --output logs/go2_him_fresh/phase3_preflight/summary.json
```

普通可视化使用：

```bash
python -m deploy.deploy_mujoco.deploy_go2 --config deploy/deploy_mujoco/configs/go2_him_fresh.yaml
```

HIM运行配置将扭转和滚动摩擦设为0，与Isaac Gym实际形状参数一致；滑动摩擦为1。
物理步长1ms，控制发布周期2ms，策略周期20ms。验收覆盖平地、0.30m/s平移、0.30rad/s行进转向、换方向、X归零、C降低和Z停止。
50秒性能测试采用与Isaac Gym一致的严格门槛：平均身高至少0.28m、最低0.23m、最大倾角0.35rad；
线速度RMSE不超过0.07m/s、偏航RMSE不超过0.12rad/s；停止后平移不超过0.03m/s、偏航不超过0.05rad/s。
短程控制流程报告仍保留原有流程门槛，必须与严格性能报告一起通过。

`keyboard`评估只覆盖实际单键接口，Q/E反向时保持前进并交换转向方向。
4000轮候选的`full`扩展评估中，后退叠加转向失败；该组合和斜移没有通过完整验证；当前按用户要求取消组合拦截，不代表已经验证这些能力。
C退出期间继续运行零速平衡策略，并逐渐切换到静态站姿，再降低。

## 上电后的现场顺序

仿真报告不替代现场记录。现场上电、连接有线网卡并准备停止终端后，先只读检查：

```bash
python -m deploy.deploy_real.deploy_real_go2 eno1 --config deploy/deploy_real/configs/go2_him_fresh.yaml --read-only --duration 5
```

之后从原厂站稳姿态进行独立站姿接管、保持和降低验收。以下命令会产生实体运动：

```bash
python -m deploy.deploy_real.go2_commission eno1 --config deploy/deploy_real/configs/go2_him_fresh.yaml --stand-check --start-standing --confirm-standing --release-mode --allow-body-movement
```

使用本次站姿验收输出的JSONL，执行零速策略接管：

```bash
python -m deploy.deploy_real.go2_commission eno1 --config deploy/deploy_real/configs/go2_him_fresh.yaml --policy-check NEW_STAND.jsonl --start-standing --confirm-standing --release-mode --allow-body-movement --allow-policy-movement
```

HIM零速流程共30秒：5秒接管、2秒保持、2秒零速策略、4秒回站姿、4秒保持、8秒降低、5秒稳定，
随后阻尼收尾。零速策略也可能迈步或漂移。使用本次通过的零速JSONL，先检查键盘入口：

```bash
python -m deploy.deploy_real.go2_keyboard --config deploy/deploy_real/configs/go2_him_fresh.yaml --baseline deploy/deploy_real/configs/go2_him_fresh_baseline.json --readiness NEW_ZERO.jsonl --mode all --check
```

现场再次确认原厂站稳和停止终端可用后，以相同参数将 `--check` 换为 `--run` 才启动。
当前HIM入口采用0.30平移/行进转向档，PI和起步恢复候选关闭。Q/E边走边转，W/S前后，A/D侧移；
X或空格归零，C受控退出，Z或Ctrl+C紧急阻尼。
C流程为制动2秒、回站姿4秒、保持4秒、降低8秒、稳定5秒，再阻尼收尾1秒。

独立停止终端：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

历史现场记录保留真实模型/配置标记，仿真不等于实机通过。当前不把旧验收记录作为键盘启动前置条件。
当前运行方式见开头；不再使用14/16Nm和额外运动阈值。
现场数据写入 `logs/go2_him_fresh/real/`，原有Go2模型和配置继续使用各自独立路径。
