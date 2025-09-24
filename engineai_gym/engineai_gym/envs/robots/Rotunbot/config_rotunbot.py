from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot


class ConfigRotunbot(ConfigLeggedRobot):
    """Configuration class for the Rotunbot spherical robot."""

    class env(ConfigLeggedRobot.env):
        num_envs = 4096
        # Observations for spherical robot
        obs_list = [
            "pendulum_angle",      #摆锤角度
            "pendulum_angular_vel", #摆锤角速度
            "base_lin_vel",        #基座线速度
            "base_ang_vel",        #基座角速度
            "projected_gravity",   #重力投影
            "actions",             #动作
        ]
        # Goals for spherical robot
        goal_list = ["commands"]
        # Joints used for action
        action_joints = ["joint1", "joint2"]  # joint1:前进后退, joint2:转向
        # Environment spacing
        env_spacing = 3.0
        # Send time out information to the algorithm
        send_timeouts = True
        # Episode length in seconds
        episode_length_s = 20

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

    class commands(ConfigLeggedRobot.commands):
        curriculum = False
        max_curriculum = 1.0
        # Time before commands are changed [s]
        resampling_time = 10.0
        # If true: compute ang vel command from heading error
        yaw_from_heading_target = True
        # Probability of still commands
        still_ratio = 0.0
        # Set command to zero if command < threshold
        lin_vel_set_zero_threshold = 0.1
        ang_vel_set_zero_threshold = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(ConfigLeggedRobot.env):
        # Initial state of the robot
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # Default joint positions
        default_joint_angles = {
            "joint1": 0.0,
            "joint2": 0.0,
        }

    class control(ConfigLeggedRobot.env):
        # PD gains for joints
        stiffness = {"joint1": 0.0, "joint2": 0.0}
        damping = {"joint1": 0.0, "joint2": 0.0}
        # Action scale for control
        action_scale = 1.0
        # Decimation factor
        decimation = 4

    class asset(ConfigLeggedRobot.env):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/Rotunbot/urdf/ball.urdf"
        name = "rotunbot"
        # Whether to use self-collisions
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
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
        track_contacts = []
        # Which links to disable collisions for
        disable_collisions = []
        # Which links are represented by a bounding box
        bounding_box_link_ids = []
        # Which links to consider for termination
        termination_contact_indices = []