from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot


class ConfigRotunbot(ConfigLeggedRobot):
    """Configuration class for the Rotunbot spherical robot."""

    class env(ConfigLeggedRobot.env):
        num_envs = 4096
        num_actions = 2
        num_observations = 26
        episode_length_s = 40  # episode length in seconds
        
        # Observations for spherical robot
        obs_list = [
            "commands",           # 命令（线速度和角速度）
            "base_quat",          # 基座四元数
            "base_lin_vel",       # 基座线速度
            "base_ang_vel",       # 基座角速度
            "last_base_lin_vel",  # 上一时刻线速度
            "last_base_ang_vel",  # 上一时刻角速度
            "dof_pos",            # 关节位置（副轴）
            "dof_vel",            # 关节速度
            "projected_gravity",  # 重力投影
            "actions",            # 动作历史
        ]
        
        # Goals for spherical robot
        goal_list = ["commands"]
        
        # Joints used for action
        action_joints = ["joint1", "joint2"]  # joint1:前进后退, joint2:转向
        
        # Environment spacing
        env_spacing = 3.0
        
        # Send time out information to the algorithm
        send_timeouts = True

    class safety(ConfigLeggedRobot.safety):
        # Torque limit multiplier for each joint
        torque_hard_limit_multi = {"joint1": 1.0, "joint2": 1.0}

    class terrain(ConfigLeggedRobot.terrain):
        # Terrain type: None, plane, heightfield or trimesh
        mesh_type = "plane"  # Start with plane terrain
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # For flat terrain only
        measure_heights = False

    class init_state(ConfigLeggedRobot.env):
        # Initial state of the robot
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # Default joint positions
        default_joint_angles = {
            "joint1": 0.0,
            "joint2": 0.0,
        }

    class control(ConfigLeggedRobot.env):
        # P and V control type
        control_type = 'P and V'  # P: position, V: velocity, T: torques
        
        # PD Drive parameters:
        stiffness = {'joint1': 20.0, 'joint2': 15.}  # [N*m/rad]
        damping = {'joint1': 5.0, 'joint2': 5.0}     # [N*m*s/rad]
        
        # Action scale parameters
        action_scale = 20
        first_actionScale = 8
        second_actionScale = 0.5236
        
        # Decimation factor: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class commands(ConfigLeggedRobot.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 2  # default: lin_vel_x, ang_vel_yaw
        # Time before commands are changed [s]
        resampling_time = 0.04
        # If true: compute ang vel command from heading error
        yaw_from_heading_target = False
        # Probability of still commands
        still_ratio = 0.0
        # Set command to zero if command < threshold
        lin_vel_set_zero_threshold = 0.1
        ang_vel_set_zero_threshold = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(ConfigLeggedRobot.domain_rands):
        randomize_base_mass = True
        added_mass_range = [-8., 8.]
        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

    class normalization(ConfigLeggedRobot.normalization):
        class obs_scales:
            lin_vel = [0.67, 3.33, 20.0]
            ang_vel = [1.25, 1.25, 1.43]
            dof_pos = 2.0
            dof_vel = [0.125, 0.4]
            command = [1.0, 2.0]
        clip_observations = 100.
        clip_actions = 50.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            quat = 0.01
            dof_pos = 0.02
            dof_vel = 0.5
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1

    class rewards(ConfigLeggedRobot.rewards):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        
        class scales:
            termination = -0.0
            tracking_lin_vel = 8.0
            tracking_ang_vel = 3.5
            lin_vel_z = -0.5
            ang_vel_xy = -0.2
            torques = -0.0008
            dof_acc = -0.01  # -2.5e-5
            action_rate = -0.2

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_lin_vel_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_ang_vel_sigma = 0.1

    class asset(ConfigLeggedRobot.env):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/Rotunbot/urdf/ball.urdf"
        name = "rotunbot"
        # Whether to use self-collisions
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # Whether to collapse fixed joints
        collapse_fixed_joints = False
        # Whether to flip visual meshes
        flip_visual_attachments = False
        # Whether to disable gravity
        disable_gravity = False
        # Whether to collapse joints
        collapse_joints = False
        # Threshold for removing links
        terminate_after_contacts = 1
        # Which links to track contacts for
        track_contacts = ["base_link"]
        # Which links to disable collisions for
        disable_collisions = []
        # Which links are represented by a bounding box
        bounding_box_link_ids = []
        # Which links to consider for termination
        termination_contact_indices = [0]  # base_link index
        
        # Asset properties for Isaac Gym
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01