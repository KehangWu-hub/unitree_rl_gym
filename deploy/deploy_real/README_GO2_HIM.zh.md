# Go2 HIM 仿真与真机验收

使用 `unitree-rl` 环境，从项目根目录运行。HIM模型位于 `deploy/pre_train/go2_him/`，
模型记录及SHA256保存在该目录的 `provenance.json` 中。替换训练权重后，需要重新完成下述仿真与部署验收。
完整模型输入270维历史，内部估计3维速度和16维隐变量，Actor输入64维、输出12维。

## 仿真与离线检查

```bash
conda activate unitree-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m deploy.deploy_real.check_go2_him
```

此检查不初始化DDS。它核对模型、历史观测、关节映射、SDK CRC、推理耗时和停止保护，
并验证仿真报告与模型、配置及控制代码的摘要一致。结果在 `logs/go2_him/phase3_preflight/summary.json`。
更换权重或修改控制代码后，需要重新生成仿真证据。

重跑完整仿真验收：先停止训练，单独完成Isaac Gym评估，再执行MuJoCo测试。

```bash
python legged_gym/scripts/evaluate_him.py --headless --num_envs 64
python -m deploy.deploy_mujoco.validate_go2_him --output logs/go2_him/phase2_validation/final_raw
python -m deploy.deploy_mujoco.validate_go2_him --protected --keys zero w s a d q e switch --output logs/go2_him/phase2_validation/final_protected
python -m deploy.deploy_mujoco.validate_go2_him --protected --stage zero --output logs/go2_him/phase2_validation/zero_stage
python -m deploy.deploy_mujoco.validate_go2_him --protected --stage forward --output logs/go2_him/phase2_validation/forward_stage
python -m deploy.deploy_mujoco.validate_go2_him --qualify
python -m deploy.deploy_real.check_go2_him
```

普通可视化使用：

```bash
python -m deploy.deploy_mujoco.deploy_go2 --config deploy/deploy_mujoco/configs/go2_him.yaml
```

HIM运行配置将扭转和滚动摩擦设为0，与Isaac Gym实际形状参数一致；滑动摩擦为1。
物理步长1ms，控制发布周期2ms，策略周期20ms。验收覆盖平地、0.30m/s平移、
0.30rad/s行进转向、换方向、X归零、C降低和Z停止。跟踪门槛是实际/指令均值至少0.5，
线速度RMSE不超过0.20m/s、偏航RMSE不超过0.25rad/s；归零后平均平移速度低于0.08m/s、
偏航低于0.15rad/s。这些是项目验收门槛，不能解释为精确速度跟踪。

## 上电后的现场顺序

仿真报告不替代现场记录。现场上电、连接有线网卡并准备停止终端后，先只读检查：

```bash
python -m deploy.deploy_real.deploy_real_go2 eno1 --config deploy/deploy_real/configs/go2_him.yaml --read-only --duration 5
```

之后从原厂站稳姿态进行独立站姿接管、保持和降低验收。以下命令会产生实体运动：

```bash
python -m deploy.deploy_real.go2_commission eno1 --config deploy/deploy_real/configs/go2_him.yaml --stand-check --start-standing --confirm-standing --release-mode --allow-body-movement
```

使用本次站姿验收输出的JSONL，执行零速策略接管：

```bash
python -m deploy.deploy_real.go2_commission eno1 --config deploy/deploy_real/configs/go2_him.yaml --policy-check NEW_STAND.jsonl --start-standing --confirm-standing --release-mode --allow-body-movement --allow-policy-movement
```

HIM零速流程共30秒：5秒接管、2秒保持、2秒零速策略、4秒回站姿、4秒保持、8秒降低、5秒稳定，
随后阻尼收尾。零速策略也可能迈步或漂移。使用本次通过的零速JSONL，先检查键盘入口：

```bash
python -m deploy.deploy_real.go2_keyboard --config deploy/deploy_real/configs/go2_him.yaml --baseline deploy/deploy_real/configs/go2_him_baseline.json --readiness NEW_ZERO.jsonl --mode all --check
```

现场再次确认原厂站稳和停止终端可用后，以相同参数将 `--check` 换为 `--run` 才启动。
当前HIM入口采用0.30平移/行进转向档，PI和起步恢复候选关闭。Q/E边走边转，W/S前后，A/D侧移；
X或空格归零，C受控退出，Z或Ctrl+C紧急阻尼。
C流程为制动2秒、回站姿4秒、保持4秒、降低8秒、稳定5秒，再阻尼收尾1秒。

独立停止终端：

```bash
python -m deploy.deploy_real.go2_watchdog --stop --domain-id 0
```

现场记录须匹配本次HIM模型与配置；旧策略记录和带 `simulation_only` 标记的结果均不能作为前置通过依据。
配置保留PD限幅14Nm、反馈停止16Nm、状态超时与姿态/关节保护；这些参数仍需在HIM现场流程中验收。
现场数据写入 `logs/go2_him/real/`，原有Go2模型和配置继续使用各自独立路径。
