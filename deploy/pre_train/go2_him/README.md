# Go2 HIM 模型说明

`policy.pt` 来自 [HIMLOCO_GO2](https://github.com/1170632760/HIMLOCO_GO2)，
对应提交 `2e70bcf66d2a479add5a688ed4d78d863d166c11`，包含完整的HIM历史编码器和Actor，权重未经修改。
详细来源与SHA256记录在 `provenance.json` 中。本地继续训练300轮的候选模型未通过验收，
本目录采用的是源仓库预训练模型。

`deploy.yaml` 记录训练与部署接口：

- 完整模型输入270维：6帧历史观测，每帧45维，最新帧在前。
- 编码器输出3维估计速度和16维归一化隐变量。
- Actor输入64维：当前观测45维＋估计速度3维＋隐变量16维。
- 模型输出12个关节位置动作，动作缩放为0.25。
- 策略PD增益为Kp＝20、Kd＝0.5。

源仓库真机配置将归一化的偏航摇杆输入乘以0.785；本项目直接使用rad/s为单位的偏航角速度指令，
采用训练时的观测缩放0.25。

MuJoCo仿真配置位于 `deploy/deploy_mujoco/configs/go2_him.yaml`，
真机配置位于 `deploy/deploy_real/configs/go2_him.yaml`，路径均相对于项目根目录。
真机部署需要为本模型重新完成现场验收，不能复用旧策略的验收记录。

源项目的CC BY-NC-SA 4.0许可证保存在 `third_party/himloco_rsl_rl/HIMLoco_LICENSE` 中；
引入的算法文件也保留了各自的原始版权和许可证声明。
