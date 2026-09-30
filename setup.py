from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='RL training, MuJoCo validation and deployment tools for Unitree Go2',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
