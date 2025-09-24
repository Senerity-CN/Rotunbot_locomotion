from engineai_gym.envs.base.config_legged_robot import ConfigLeggedRobot


class ConfigA1Rough(ConfigLeggedRobot):
    class env(ConfigLeggedRobot.env):
        action_joints = [
            "FL_hip_joint",
            "RL_hip_joint",
            "FR_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "RL_thigh_joint",
            "FR_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "RL_calf_joint",
            "FR_calf_joint",
            "RR_calf_joint",
        ]

    class init_state(ConfigLeggedRobot.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(ConfigLeggedRobot.control):
        control_type = "P"
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scales = {"joint": 0.25}
        decimation = 4

    class asset(ConfigLeggedRobot.asset):
        file = "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/quadruped/a1/urdf/a1.urdf"
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        joint_armature = {"hip": 0.001, "thigh": 0.001, "calf": 0.001}
        joint_friction = {"hip": 0, "thigh": 0, "calf": 0}

    class rewards(ConfigLeggedRobot.rewards):
        class params(ConfigLeggedRobot.rewards.params):
            soft_dof_pos_limit_multi = 0.9
            base_height_target = 0.25
            soft_dof_torque_limit_multi = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}

        class scales(ConfigLeggedRobot.rewards.scales):
            feet_distance = 0.0
            foot_slip = 0.0
            base_acc = 0.0
            vel_mismatch_exp = 0.0
            track_vel_hard = 0.0
            default_joint_pos = 0.0
            low_speed = 0.0
            action_smoothness = -0.0
            torques = -0.0002
            dof_pos_limits = -10.0
