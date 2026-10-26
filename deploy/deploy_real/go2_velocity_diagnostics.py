"""Read-only LowState/SportModeState speed-source inspection; no motor publisher."""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from deploy.deploy_real.deploy_real_go2 import Go2RealConfig
from deploy.deploy_real.go2_commission import VerifiedTransport
from deploy.deploy_real.go2_keyboard import CONFIG, ROOT


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="eno1")
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Inspect SDK fields offline; no DDS initialization")
    parser.add_argument("--analyze", type=Path, help="离线检查键盘shadow日志的接管后速度连续性")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not 1 <= args.duration <= 60:
        parser.error("Read-only duration must be 1..60 seconds")
    if args.analyze:
        if args.check:
            parser.error('--analyze与--check不能同时使用')
        from deploy.deploy_real.go2_keyboard import BASELINE
        from deploy.deploy_real.go2_velocity_feedback import VelocityFeedback, validate_feedback_readiness
        manifest = json.loads(BASELINE.read_text())
        try:
            rows = [json.loads(line) for line in args.analyze.read_text().splitlines() if line.strip()]
            observation = next(r['feedback']['velocity_feedback'] for r in rows
                if r.get('mode') == 'policy' and 'velocity_feedback' in r.get('feedback',{}))
            source = observation.get('source','sport')
            profile = observation.get('speed_profile','legacy')
            config = Go2RealConfig.load(CONFIG, str(ROOT))
            observer = VelocityFeedback('shadow', [.5,.4,.5] if profile == 'uniform' else config.data['command_limits'],
                'body' if source == 'kinematic' else observation.get('frame') if observation.get('frame') in ('body','world') else None,
                source=source, config=config, speed_profile=profile)
            result = validate_feedback_readiness(args.analyze, manifest['policy_sha256'], manifest['config_sha256'],
                source=source, estimator_sha256=getattr(observer.source,'fingerprint',None),
                speed_profile=profile if profile == 'uniform' else None,
                profile_sha256=observer.profile_sha256 if profile == 'uniform' else None)
            result['source'] = source
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            print(json.dumps({'publication_ready': False, 'reason': str(error)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps({'publication_ready': True, **result}, ensure_ascii=False, indent=2))
        return 0
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    if args.check:
        print(json.dumps({"read_only": True, "dds_initialized": False,
                          "sport_state_fields": list(SportModeState_.__annotations__),
                          "scope": "Field availability does not prove live publication, frame or estimation accuracy."}, indent=2))
        return 0
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    config = Go2RealConfig.load(CONFIG, str(ROOT))
    ChannelFactoryInitialize(config.data["dds"]["domain_id"], args.network)
    low = VerifiedTransport(config, publish=False)
    sport_rows, low_rows = [], []
    lock = threading.Lock()

    def callback(message):
        row = {"received_monotonic": time.monotonic(),
               "stamp": {"sec": int(message.stamp.sec), "nanosec": int(message.stamp.nanosec)},
               "error_code": int(message.error_code), "mode": int(message.mode),
               "velocity_raw": list(map(float, message.velocity)),
               "position_raw": list(map(float, message.position)),
               "yaw_speed_raw": float(message.yaw_speed),
               "quaternion_wxyz": list(map(float, message.imu_state.quaternion)),
               "foot_force_raw": list(map(int, message.foot_force)),
               "foot_position_body_raw": list(map(float, message.foot_position_body)),
               "foot_speed_body_raw": list(map(float, message.foot_speed_body))}
        with lock:
            sport_rows.append(row)

    subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    subscriber.Init(callback, 10)
    print("只读采集 LowState 与 rt/sportmodestate；不发布电机命令、不切换模式。", flush=True)
    start = time.monotonic()
    last_state = None
    while time.monotonic() - start < args.duration:
        state = low.snapshot()
        if state is not None and state.timestamp != last_state:
            last_state = state.timestamp
            low_rows.append({"received_monotonic": state.timestamp,
                "quaternion_wxyz": state.quaternion_wxyz.tolist(),
                "angular_velocity": state.angular_velocity.tolist(),
                "joint_pos_sdk": state.joint_pos_sdk.tolist(),
                "joint_vel_sdk": state.joint_vel_sdk.tolist(),
                "foot_force_raw": (state.feedback or {}).get("foot_force_raw"),
                "state_tick": (state.feedback or {}).get("state_tick")})
        time.sleep(.02)
    with lock:
        sport = list(sport_rows)
    summary = {"read_only": True, "network": args.network, "duration_seconds": time.monotonic() - start,
               "low_state_samples": len(low_rows), "sport_samples": len(sport),
               "sport_velocity_frame": "unverified", "sport_speed_accuracy_verified": False,
               "scope": "Raw SDK estimates only. Verify publication/freshness after control handoff, coordinate frame, and independent speed accuracy before closing a feedback loop."}
    if sport:
        timestamps = np.array([r["received_monotonic"] for r in sport])
        velocity = np.array([r["velocity_raw"] for r in sport])
        summary.update(sport_receipt_hz=(len(sport) - 1) / (timestamps[-1] - timestamps[0]) if len(sport) > 1 else None,
                       unique_sport_stamps=len({(r["stamp"]["sec"], r["stamp"]["nanosec"]) for r in sport}),
                       velocity_all_finite=bool(np.isfinite(velocity).all()),
                       velocity_min_raw=velocity.min(axis=0).tolist(), velocity_max_raw=velocity.max(axis=0).tolist(),
                       velocity_mean_raw=velocity.mean(axis=0).tolist())
    output = args.output or ROOT / "logs/go2_real" / f"velocity_readonly_{time.time_ns()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(json_safe({"summary": summary, "sport_state": sport, "low_state": low_rows}), stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print(f"Output: {output}")
    return 0 if sport and low_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
