# Go2 HIM 模型说明

本目录保存本项目使用的Go2 HIM部署模型与接口配置。
`policy.pt` 包含完整的历史编码器和Actor，`deploy.yaml` 定义与模型配套的观测、动作和控制参数。

`deploy.yaml` 记录训练与部署接口：

- 完整模型输入270维：6帧历史观测，每帧45维，最新帧在前。
- 编码器输出3维估计速度和16维归一化隐变量。
- Actor输入64维：当前观测45维＋估计速度3维＋隐变量16维。
- 模型输出12个关节位置动作，动作缩放为0.25。
- 策略PD增益为Kp＝20、Kd＝0.5。

偏航角速度指令以rad/s为单位，观测缩放为0.25。

MuJoCo仿真配置位于 `deploy/deploy_mujoco/configs/go2_him.yaml`，
真机配置位于 `deploy/deploy_real/configs/go2_him.yaml`，路径均相对于项目根目录。
真机部署需要为本模型重新完成现场验收，不能复用旧策略的验收记录。

后续训练完成后，导出完整的HIM模型，替换 `policy.pt` 及对应接口配置，
同步更新 `provenance.json` 中的训练记录与模型摘要，并重新执行仿真和现场验收。
训练、导出及验收命令见 [HIM部署指南](../../deploy_real/README_GO2_HIM.zh.md)。
