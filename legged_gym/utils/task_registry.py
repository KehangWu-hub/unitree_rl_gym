import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # 复制随机种子
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """根据已注册的任务名称或传入的配置创建环境。

        参数：
            name (string)：已注册环境的名称。
            args (Args，可选)：Isaac Gym命令行参数。为None时调用get_args()，默认为None。
            env_cfg (Dict，可选)：用于覆盖已注册配置的环境配置，默认为None。

        异常：
            ValueError：找不到与name对应的已注册环境时抛出。

        返回：
            isaacgym.VecTaskPython：创建的环境。
            Dict：对应的配置。
        """
        # 未传入参数时读取命令行参数
        if args is None:
            args = get_args()
        # 检查是否存在该名称对应的已注册环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # 加载配置
            env_cfg, _ = self.get_cfgs(name)
        # 使用命令行参数覆盖配置中的对应项（如果指定）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # 解析仿真参数（先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """根据已注册的任务名称或传入的配置创建训练算法运行器。

        参数：
            env (isaacgym.VecTaskPython)：用于训练的环境（待办：从算法内部移除该依赖）。
            name (string，可选)：已注册环境的名称。为None时改用传入的配置，默认为None。
            args (Args，可选)：Isaac Gym命令行参数。为None时调用get_args()，默认为None。
            train_cfg (Dict，可选)：训练配置。为None时根据name获取配置，默认为None。
            log_root (str，可选)：TensorBoard日志目录。设为None时不记录日志，例如测试阶段。
                                      日志保存到<日志根目录>/<日期时间>_<运行名称>；默认目录为
                                      <LEGGED_GYM路径>/logs/<实验名称>。

        异常与警告：
            ValueError：name和train_cfg均未提供时抛出。
            Warning：name和train_cfg同时提供时忽略name。

        返回：
            PPO：创建的算法运行器。
            Dict：对应的训练配置。
        """
        # 未传入参数时读取命令行参数
        if args is None:
            args = get_args()
        # 优先使用传入的训练配置，否则根据任务名称加载配置
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载配置
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # 使用命令行参数覆盖配置中的对应项（如果指定）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        # 创建新日志目录前保留续训设置
        resume = train_cfg.runner.resume
        if resume:
            # 加载此前训练的模型
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# 创建全局任务注册器
task_registry = TaskRegistry()
