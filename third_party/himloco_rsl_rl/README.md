# HIMLoco dependency

Source: [1170632760/HIMLOCO_GO2](https://github.com/1170632760/HIMLOCO_GO2),
commit `2e70bcf66d2a479add5a688ed4d78d863d166c11`.
Its 19 `rsl_rl` Python files match [InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco)
commit `ef289acaa62795009363b7b819c9186690630441` byte for byte before adaptation.
Original hashes are recorded in `upstream.json`.

Local changes:

- Absolute imports use `third_party.himloco_rsl_rl.rsl_rl`; no global install,
  `sys.path` override, or replacement of the existing `rsl_rl` dependency.
- HIM runner checkpoints record the completed iteration at each save, load on
  the requested device, and allow inference with no log directory. Final losses
  are retained for validation; optimizer learning rates are restored as well.
- Trailing whitespace is removed from imported Python files.

The policy, estimator, contrastive objective, PPO and storage are retained from
upstream. Preserve the included BSD notices and the original project's
`HIMLoco_LICENSE` (CC BY-NC-SA 4.0).

The environment/config adaptations live in `legged_gym/envs/go2_him`:
imports/task names, an adapter for the seven-value step interface, clearing
history and refreshing body-frame observations at episode reset, and respecting
the noise-disable flag for terrain heights during playback. The Go2 model
is the repository's existing URDF, rather than a second copy of the asset tree.
