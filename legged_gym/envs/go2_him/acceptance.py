"""Performance gates, separate from training rewards and runtime fall protection."""
import numpy as np

THRESHOLDS = dict(mean_height_min=.28, min_height=.23, tilt_max=.35,
                  linear_rmse_max=.07, yaw_rmse_max=.12,
                  stopped_speed_max=.03, stopped_yaw_max=.05, action_bound=4.)


def assess_motion(samples, reset_count=0, action_max=0., thresholds=None):
    """Columns: time, command xyz, measured vx/vy/yaw, height, tilt.

    Accepts a single trajectory or [time, environments, columns]; evaluates the
    worst environment so averaging cannot hide a stalled leg or a fall.
    """
    limits=dict(THRESHOLDS if thresholds is None else thresholds)
    data=np.asarray(samples)
    if data.ndim==2:data=data[:,None,:]
    if data.ndim!=3 or data.shape[-1]!=9 or not len(data) or not np.isfinite(data).all():
        return {'passed':False,'reason':'invalid_or_nonfinite_telemetry'}
    # Exclude two seconds after command transitions, but retain transient tilt/fall checks.
    t=data[:,0,0]
    moving=((t>=2)&(t<20))|((t>=22)&(t<40))
    stopped=t>=42
    if not moving.any() or not stopped.any() or t[-1]<49.9:
        return {'passed':False,'reason':'requires_complete_50_second_trial'}
    error=data[moving,:,4:7]-data[moving,:,1:4]
    rmse=np.sqrt(np.mean(error**2,axis=0)).max(axis=0)
    minimum_mean_height=float(data[moving,:,7].mean(axis=0).min())
    minimum_height=float(data[:,:,7].min());tilt=float(data[:,:,8].max())
    residual=float(np.linalg.norm(data[stopped,:,4:6],axis=-1).mean(axis=0).max())
    residual_yaw=float(np.abs(data[stopped,:,6]).mean(axis=0).max())
    checks={'no_resets':reset_count==0,'height_mean':minimum_mean_height>=limits['mean_height_min'],
            'height_min':minimum_height>=limits['min_height'],'tilt':tilt<=limits['tilt_max'],
            'linear_tracking':bool(np.all(rmse[:2]<=limits['linear_rmse_max'])),
            'yaw_tracking':bool(rmse[2]<=limits['yaw_rmse_max']),
            'stopped_speed':residual<=limits['stopped_speed_max'],
            'stopped_yaw':residual_yaw<=limits['stopped_yaw_max'],
            'action_bound':np.isfinite(action_max) and action_max<=limits['action_bound']}
    return {'passed':all(checks.values()),'checks':checks,'rmse':rmse.tolist(),
            'minimum_environment_mean_height':minimum_mean_height,'min_height':minimum_height,
            'max_tilt':tilt,'stopped_speed':residual,'stopped_yaw':residual_yaw,
            'reset_count':int(reset_count),'action_max':float(action_max),'thresholds':limits}
