"""Compare two HIM policies with identical MuJoCo/controller settings; no DDS."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from deploy.common.him_qualification import sha256
from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_mujoco.validate_go2_him import ROOT
# Keep these names importable for existing experiment scripts.
from deploy.deploy_mujoco.gait_metrics import COMMANDS, LEGS, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/deploy_real/configs/go2_him_fresh.yaml')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='+', choices=tuple(COMMANDS), default=list(COMMANDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = Go2RealConfig.load(args.config, str(ROOT))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'simulation_only': True, 'dds_initialized': False, 'leg_order': LEGS,
              'config_sha256': sha256(args.config), 'software_motion_limits_enabled': config.software_motion_limits_enabled,
              'scope': 'Identical controller and initial state; 0.30 commands, 5-second steady foot metrics. '
                       '5N contact and 40ms flight are analysis labels, not robot limits. '
                       'No completed flight is reported as null, never as good clearance.', 'policies': {}}
    for name, path in (('v1', config.policy_path), ('v2', args.candidate.resolve())):
        cfg = replace(config, policy_path=str(path))
        case = {'path': str(path), 'sha256': sha256(path), 'cases': {}}
        report['policies'][name] = case
        for key in args.keys:
            case['cases'][key] = measure(cfg, key, args.output/f'{name}_{key}')
            (args.output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({'policy': name, 'key': key, **case['cases'][key]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
