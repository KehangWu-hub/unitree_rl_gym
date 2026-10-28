"""Pinned Go2 adapter recipe with flat terrain as the explicit experiment variable.

Inherit the upstream Gaussian policy, optimizer, rewards, command sampler and
randomization. Correctness fixes in Go2HIMRobot remain in place. This is an
offline baseline, not a hardware-qualified policy.
"""
from .go2_him_config import Go2HIMCfg, Go2HIMCfgPPO


class Go2HIMUpstreamFlatCfg(Go2HIMCfg):
    training_profile = 'go2_him_upstream_flat_v1'
    training_task = 'go2_him_upstream_flat'
    action_distribution = 'gaussian'
    hardware_deployment_blocked = True

    class terrain(Go2HIMCfg.terrain):
        mesh_type = 'plane'
        curriculum = False


class Go2HIMUpstreamFlatCfgPPO(Go2HIMCfgPPO):
    class runner(Go2HIMCfgPPO.runner):
        experiment_name = 'go2_him_upstream_flat'
