# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os, sys
from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

##
# Scene settings
##


@configclass
class TableCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    # usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/table_instanceable.usd")


@configclass
class BackgroundCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/background_instanceable.usd")

@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/cube_instanceable.usd"),
        # usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/cube_instanceable_mass.usd"),
        # usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/cylinder_small_instanceable.usd"),
        scale=(1, 1, 1),
        # scale=(0.8, 0.8, 0.8),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        # static_friction=4., dynamic_friction=4., restitution=0.0, prim_path="/World/Materials/cubeMaterial"
        static_friction=1., dynamic_friction=1., restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )


@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        # position_uniform_min = [0.50, -0.05, 0.04]  # position (x,y,z)
        # position_uniform_max = [0.55, 0.05, 0.04]  # position (x,y,z)
        position_uniform_min = [0.45, -0.1, 0.04]  # position (x,y,z)
        position_uniform_max = [0.55, 0.1, 0.04]  # position (x,y,z)

    @configclass
    class ObjectDesiredPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.55, 0.0, 0.06]  # position default (x,y,z)
        # position_default = [0.6, 0.0, 0.2]  # position default (x,y,z)
        position_uniform_min = [0.55, -0.05, 0.08]  # position (x,y,z)
        position_uniform_max = [0.6, 0.05, 0.08]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    object_material_properties = {
        "enabled": True,
        "static_friction_range": (0.95, 1.05),
        "dynamic_friction_range": (0.95, 1.05),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 5,
    }

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        # -- joint state
        # arm_dof_pos = {"scale": 1.0}
        # arm_dof_pos_scaled = {"scale": 1.0}
        # arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        # tool_positions = {"scale": 1.0}
        tool_positions = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.003, "max": 0.003}}
        # tool_orientations = {"scale": 1.0}
        # -- object state
        # object_positions = {"scale": 1.0}
        object_positions = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.003, "max": 0.003}}
        # object_orientations = {"scale": 1.0}
        object_orientations = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.003, "max": 0.003}}
        object_relative_tool_positions = {"scale": 1.0}
        is_grasped = {"scale": 1.0}
        # object_relative_tool_orientations = {"scale": 1.0}
        # -- object desired state
        # object_to_goal_positions = {"scale": 1.0, }
        object_to_goal_positions = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.003, "max": 0.003}}
        tool_actions_bool = {'scale': 1.0}
        # ee_actions = {'scale': 1.0}
        # object_desired_positions = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = True
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- robot-centric
    # reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.15}
    # reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.25}
    reaching_object_position_tanh = {"weight": 1., "sigma": 6}
    # opening_gripper = {'weight': 0.01}
    # tracking_object_position_tanh = {"weight": 5., "sigma": 0.2}
    tracking_object_position_tanh = {"weight": 2., "sigma": 2}
    # penalizing_action_rate_l2 = {"weight": 0.05}
    grasp_object_success = {'weight': 1.}
    # lifting_object_success = {"weight": 3.25, "threshold": 0.08}
    lifting_object_success = {"weight": 4., "threshold": 0.08}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    object_falling = True  # reset when object falls off the table
    is_success = False  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "inverse_kinematics"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    moving_average = False
    decay = 0.7

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="position_rel",
        ik_method="dls",
        position_command_scale=(0.02, 0.02, 0.02),
        rotation_command_scale=(0.05, 0.05, 0.05),
        ee_min_limit=(0.3, -0.2, 0.0),
        ee_max_limit=(0.65, 0.2, 0.3)
    )

@configclass
class DomainRandomizationCfg:
    randomize = False
    every_step = True
    perlin_noise = True
    randomize_object = True
    randomize_table = True
    randomize_light = True
    randomize_robot = True
    randomize_background = True
    randomize_camera = True
    camera_pos_noise = 0.01
    camera_ori_noise = 0.03
    random_obs_amplitude = False
    randomize_action = True

##
# Environment configuration
##


@configclass
class LiftEnvCfg(IsaacEnvCfg):
    """Configuration for the Lift environment."""

    # General Settings
    # env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=2 * (1/100) * 150)
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=2 * (1/250) * 150)
    viewer: ViewerCfg = ViewerCfg(debug_vis=False, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=1/250,
        substeps=5,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene Settings
    # -- robot
    robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # -- table
    table: TableCfg = TableCfg()
    background: BackgroundCfg = BackgroundCfg()
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
