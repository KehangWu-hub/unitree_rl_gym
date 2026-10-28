# 环境安装

[项目首页](../README.md) · [HIM指南](../deploy/deploy_real/README_GO2_HIM.zh.md) · [PPO部署](../deploy/deploy_real/README_GO2.zh.md)

所有命令均以仓库根目录为工作目录。训练使用NVIDIA GPU和Isaac Gym Preview 4；只运行MuJoCo或部署已有模型时，无需安装Isaac Gym。

## 依赖分工

| 用途 | 依赖 |
|---|---|
| 模型推理与MuJoCo | PyTorch、NumPy、PyYAML、MuJoCo |
| Isaac Gym训练 | Isaac Gym、rsl-rl、TensorBoard、Matplotlib |
| 真机通信 | unitree_sdk2py及其DDS依赖 |

当前项目运行环境版本如下，可作为复现参考：

| 组件 | 版本 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.13.1＋CUDA 11.7 |
| NumPy | 1.23.5 |
| MuJoCo | 3.2.3 |
| Isaac Gym | Preview 4（1.0rc4） |
| rsl-rl | 1.0.2 |

## 基础环境

```bash
git clone https://github.com/KehangWu-hub/unitree_rl_gym.git
cd unitree_rl_gym
conda create -n unitree-rl python=3.8 -y
conda activate unitree-rl

python -m pip install torch==1.13.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy==1.23.5 mujoco==3.2.3 pyyaml==6.0.3
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

其他平台的PyTorch安装方式见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

安装依赖后，可直接通过 `python -m` 运行仓库中的部署模块。MuJoCo的图形窗口需要可用的桌面/OpenGL环境；服务器可使用 `--headless`。

## 训练依赖

从[NVIDIA Isaac Gym页面](https://developer.nvidia.com/isaac-gym)获取Preview 4，解压后安装；将下列路径替换为实际解压位置：

```bash
python -m pip install -e /path/to/isaacgym/python
python -m pip install tensorboard==2.12.0 matplotlib==3.7.5
python -m pip install 'git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2'
```

HIM算法随本仓库保存在 `third_party/himloco_rsl_rl/`；外部 `rsl-rl` 用于原PPO及训练工具接口，两者并存。

在项目根目录注册可编辑安装：

```bash
python -m pip install -e . --no-deps
```

这里使用 `--no-deps`，按上面的已选版本管理依赖，避免旧 `setup.py` 中的NumPy约束覆盖当前环境。

检查训练依赖时先导入Isaac Gym：

```bash
python -c "import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())"
```

若启动时提示找不到 `libpython3.8.so`，在当前终端补充Conda动态库路径：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 真机通信依赖

按[Unitree SDK2 Python安装说明](https://github.com/unitreerobotics/unitree_sdk2_python#installation)配置Cyclone DDS并安装SDK：

```bash
python -m pip install 'git+https://github.com/unitreerobotics/unitree_sdk2_python.git'
```

检查导入：

```bash
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK2 ready')"
```

随后按[HIM指南](../deploy/deploy_real/README_GO2_HIM.zh.md)或[PPO指南](../deploy/deploy_real/README_GO2.zh.md)运行。终端重新打开后，需要重新激活Conda环境并设置项目的 `PYTHONPATH`。
