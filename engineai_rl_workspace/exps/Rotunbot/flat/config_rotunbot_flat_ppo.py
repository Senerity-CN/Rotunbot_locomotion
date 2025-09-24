from engineai_rl.algos.ppo import ConfigPpo


class ConfigRotunbotFlatPpo(ConfigPpo):
    """PPO configuration for Rotunbot on flat terrain."""

    class runner(ConfigPpo.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1500
        # logging
        save_interval = 50
        experiment_name = 'rotunbot_flat'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1

    class policy(ConfigPpo.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(ConfigPpo.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class params(ConfigPpo.params):
        # Custom parameters for spherical robot
        pass