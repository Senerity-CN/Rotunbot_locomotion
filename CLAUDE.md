# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the EngineAI RL Workspace, a versatile and universal RL framework for legged robots. It includes environment, training, and evaluation components for RL algorithms with a modular design for extensibility.

Key components:
- `engineai_rl`: Contains RL algorithms implementations (PPO, etc.)
- `engineai_gym`: Contains environment implementations for robots using Isaac Gym
- `engineai_rl_lib`: Supporting library functions
- `engineai_rl_workspace`: Main workspace with training scripts and experiment configurations

## Common Development Commands

### Installation
```bash
# Install the core packages
pip install -e engineai_rl
pip install -e engineai_gym
pip install -e engineai_rl_lib
pip install -e .

# Or install all dependencies directly
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 numpy astor matplotlib pygame MNN onnx redis pre-commit tensorboard wandb isaacgym moviepy
```

### Training
```bash
# Basic training command (from engineai_rl_workspace directory)
python engineai_rl_workspace/scripts/train.py --exp_name your_experiment_name

# Resume training from a checkpoint
python engineai_rl_workspace/scripts/train.py --exp_name your_experiment_name --resume

# Multi-GPU training
torchrun --nproc_per_node=NUM_GPUS engineai_rl_workspace/scripts/train.py --exp_name your_experiment_name
```

### Code Quality
```bash
# Run pre-commit hooks (formatting, linting)
pre-commit run --all-files

# Run black formatter
black .

# Run flake8 linter
flake8 .
```

## Code Architecture

### High-Level Structure
1. **engineai_rl**: Implements reinforcement learning algorithms
   - `algos/`: Algorithm implementations (PPO, etc.)
   - `modules/`: Network modules and components
   - `runners/`: Training loop implementations
   - `storage/`: Data storage for training
   - `wrapper/`: Environment wrappers

2. **engineai_gym**: Robot environment implementations
   - `envs/`: Environment definitions for different robots
   - `resources/`: Robot models and assets
   - `tester/`: Testing utilities
   - `utils/`: Utility functions
   - `wrapper/`: Environment wrappers

3. **engineai_rl_lib**: Supporting library functions
   - Git operations, JSON handling, Redis locks, etc.

4. **engineai_rl_workspace**: Main workspace
   - `scripts/train.py`: Main training script
   - `exps/`: Experiment configurations
   - `utils/`: Utility functions for the workspace

### Key Design Patterns
- Modular design with separate modules for Env Obs, Domain Rands, Goals, Rewards, Algo Runners, Algos, Networks, and Storage
- Shared runner logic for training and evaluation
- Inheritance-based configuration system for experiments
- Automatic code state tracking and restoration for reproducible runs
- Multi-GPU support for faster training

### Training Workflow
1. Configuration is defined in experiment files in `engineai_rl_workspace/exps/`
2. The `train.py` script initializes the environment and algorithm
3. Runners handle the training loop
4. Models are saved periodically with automatic checkpointing
5. TensorBoard logs are generated for monitoring