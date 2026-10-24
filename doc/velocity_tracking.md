# Go2 速度观测与补偿

## 直接查看仿真速度

已有 MuJoCo 入口可以读取速度，无需独立评估框架：

```bash
python deploy/deploy_mujoco/deploy_go2.py \
  --headless --no-realtime --duration 15 --command 0.3 0.0 0.0
```

结束后，终端输出 `mean_body_velocity`、`velocity_error_xy`、`mean_yaw_rate` 和 `yaw_rate_error`。前两项分别为机身坐标系平均线速度及其与指令的差；偏航角速度来自 IMU。平均值包含当前运行记录的过渡阶段，不能当作专门选定的稳态窗口，也不能替代逐周期曲线。

实现见 [deploy_go2.py](../deploy/deploy_mujoco/deploy_go2.py)：`rotation_world_from_body.T @ data.qvel[:3]` 将仿真世界坐标系线速度转换到机身坐标系。当前是在结束后输出统计，没有在窗口中实时叠加速度对照。

## 保留的反馈实现

[go2_velocity_control.py](../deploy/common/go2_velocity_control.py) 保存供学习和研究复用的控制实现：

- `VectorVelocityPI.update()`：目标渐变、速度滤波、比例积分补偿、积分抗饱和及输出限幅。
- `stance_velocity_damping()`：利用仿真速度与站立足接触信息，通过雅可比计算有界修正力矩。

实验采用的 PI 参数保存在 [velocity_tracking.yaml](../deploy/deploy_mujoco/configs/velocity_tracking.yaml)：Kp=0.8、Ki=0.8、滤波时间常数0.1秒。构造函数的默认滤波值为0.2秒，复用时应显式传入实验参数。内部补偿范围来自冻结策略的训练导出，不是宇树硬件速度规格。

[contact_alignment.yaml](../deploy/deploy_mujoco/configs/contact_alignment.yaml) 保存接触参数对齐的研究配置，不替换普通入口的 `configs/go2.yaml`。

这些实现和参数仅作为仿真研究对照保留，普通 MuJoCo 与真机入口不会自动启用。它们使用的仿真速度和接触真值不能直接作为真机反馈来源。

已完成的一次性批量评估、对照诊断和报告生成工具归档到本地 `logs/archive/`。`logs/go2_tracking/` 保留最终验收、无补偿基线、已有仅 PI 实验及接触参数和失败诊断数据；已删除被最终验收替代的调参批次轨迹与重复源码，其结果报告集中保存在 `20261022_cleanup_index.json`。这些目录不随 Git 上传。后续改进策略应回到 Isaac 训练环境分析奖励权衡、指令采样和训练收敛。
