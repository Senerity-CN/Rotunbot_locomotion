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

# Training with specific parameters
python engineai_rl_workspace/scripts/train.py --exp_name rotunbot_flat_ppo --sub_exp_name test_run --num_envs 4096 --max_iterations 1500
```

### Testing/Playing
```bash
# Run a trained policy
python engineai_rl_workspace/scripts/play.py --exp_name your_experiment_name --resume

# Run with joystick control
python engineai_rl_workspace/scripts/play.py --exp_name your_experiment_name --resume --use_joystick

# Run with video recording
python engineai_rl_workspace/scripts/play.py --exp_name your_experiment_name --resume --video --record_length 500
```

### Policy Export
```bash
# Export policy to ONNX and MNN formats
python engineai_rl_workspace/scripts/export_policy.py --exp_name your_experiment_name --resume
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
   - `scripts/play.py`: Policy execution script
   - `scripts/export_policy.py`: Policy export script
   - `exps/`: Experiment configurations
   - `utils/`: Utility functions for the workspace

### Key Design Patterns
- Modular design with separate modules for Env Obs, Domain Rands, Goals, Rewards, Algo Runners, Algos, Networks, and Storage
- Shared runner logic for training and evaluation
- Inheritance-based configuration system for experiments
- Automatic code state tracking and restoration for reproducible runs
- Multi-GPU support for faster training
- Flexible observation and action handling through wrapper classes

### Training Workflow
1. Configuration is defined in experiment files in `engineai_rl_workspace/exps/`
2. The `train.py` script initializes the environment and algorithm
3. Runners handle the training loop
4. Models are saved periodically with automatic checkpointing
5. TensorBoard logs are generated for monitoring

### Key Components

#### Environment Architecture
- **LeggedRobot**: Base class for all legged robot environments
- **Obs**: Observation handling with various observation types
- **Rewards**: Reward computation with modular reward components
- **DomainRands**: Domain randomization for robust training
- **Goals**: Goal generation for target-based tasks

#### Algorithm Architecture
- **PPO**: Proximal Policy Optimization implementation
- **ActorCritic**: Actor-critic network architecture
- **MLP**: Multi-layer perceptron networks
- **RolloutStorage**: Storage for experience replay

#### Workspace Architecture
- **ExpRegistry**: Experiment registration and management
- **InputRetrivalEnvWrapper**: Environment wrapper for input processing
- **OnPolicyRunner**: Runner for on-policy algorithms

### Command Line Arguments
The framework supports extensive command-line arguments for customization:
- `--exp_name`: Experiment name for training/testing
- `--resume`: Resume training from checkpoint
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations
- `--headless`: Run without GUI
- `--video`: Record training/playback videos
- `--use_joystick`: Use joystick for manual control during playback