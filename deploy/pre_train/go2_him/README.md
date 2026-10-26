# Go2 HIM model

`policy.pt` is the unmodified complete HIM encoder and Actor supplied by
[HIMLOCO_GO2](https://github.com/1170632760/HIMLOCO_GO2), commit
`2e70bcf66d2a479add5a688ed4d78d863d166c11`. Exact provenance and SHA256 are in
`provenance.json`. The local 300-iteration fine-tuning candidate failed validation
and is not the model in this directory.

`deploy.yaml` records the training interface: six newest-first frames of 45 values,
three estimated velocities, a normalized 16-value latent, a 64-input Actor, and
12 position actions. Nominal PD gains are 20/0.5 and action scale is 0.25.
The original real-controller YAML scales the normalized yaw joystick by 0.785;
this project supplies yaw in rad/s and uses the training observation scale 0.25.

Simulation and field configurations are in `deploy/deploy_mujoco/configs/go2_him.yaml`
and `deploy/deploy_real/configs/go2_him.yaml`. Physical readiness must be recorded
for this model; previous policies' records cannot be reused.

Upstream project licensing is retained in
`third_party/himloco_rsl_rl/HIMLoco_LICENSE` (CC BY-NC-SA 4.0); vendored algorithm
files also retain their original notices.
