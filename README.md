# MBPO Minimal Reproduction (Pendulum-v1)

## 项目概述

This is a minimal, runnable MBPO-style implementation for learning and experimentation.
It includes:

- Soft Actor-Critic (SAC) policy learning
- Ensemble dynamics model
- Short model rollouts to generate synthetic transitions
- Mixed real/model replay updates

## 安装说明

```bash
git clone https://github.com/xiaoshengdianzi/MBPO_Project
cd MBPO_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 使用方法

### 训练命令

```bash
python train_mbpo.py --device cpu --steps 120000
```

### 模块化版本

```bash
python train.py --env_name Pendulum-v1 --num_episodes 20
```

### 参数说明

- `--rollout_horizon 1|3|5`
- `--rollout_freq 250`
- `--updates_per_step 1`
- `--eval_interval 5000`
- `--seed 42`

## 项目结构

```
06_MBPO/
│
├─ mbpo/                # 主要功能模块包
│   ├─ buffer.py        # 经验回放池（ReplayBuffer）
│   ├─ dynamics.py      # 环境动力学模型与 FakeEnv
│   ├─ mbpo.py          # MBPO 主流程（训练循环、模型推演等）
│   ├─ sac.py           # SAC 算法与神经网络结构
│   └─ __init__.py      # 包初始化
│
├─ train.py             # 主入口脚本，负责参数解析、训练与画图
├─ requirements.txt     # 依赖包清单
└─ README.md            # 项目说明与用法
```

各模块功能简述：

- mbpo/buffer.py：实现经验回放池 ReplayBuffer，用于存储和采样交互数据。
- mbpo/dynamics.py：实现环境动力学模型（集成网络）和 FakeEnv，用于模型推演生成虚拟样本。
- mbpo/sac.py：实现 SAC 算法的策略网络、Q 网络及其优化器。
- mbpo/mbpo.py：实现 MBPO 主流程，包括模型训练、数据混合、主训练循环等。
- train.py：主入口，负责参数解析、环境初始化、训练流程和结果可视化。

## 结果展示

![MBPO Training Return](mbpo_return.png)

## 预期行为

For `Pendulum-v1`, average return should improve over training steps.
The exact final return depends on seed and machine speed.

## 注意事项

- This is designed for clarity over full benchmark parity.
- To reproduce MuJoCo benchmark numbers, extend this code with:
  - uncertainty-aware model selection,
  - elite model filtering,
  - adaptive rollout horizon schedule,
  - benchmark environments and tuned hyperparameters.