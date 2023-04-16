# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os, sys

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.controllers.inverse_kinematics import InverseKinematicsCfg
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
    usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/table_instanceable.usd")


@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/cube_instanceable.usd"),
        scale=(1., 1., 1.),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.045), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
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
        static_friction=1.6, dynamic_friction=1.4, restitution=0.0, prim_path="/World/Materials/cubeMaterial",
    )

@configclass
class GoalMarkerCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path = os.path.join(os.environ['ORBIT_PATH'], "source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/push/assets/goal_marker_instanceable.usd"),
        scale=(1, 1, 1),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.2, 0.0), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        disable_gravity=True,
    )
    collision_props = RigidObjectCfg.CollisionPropertiesCfg(
        collision_enabled=False,
    )


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
        # position_cat: str = "uniform"  # randomize position: "default", "uniform"
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.45, 0.0, 0.045]  # position default (x,y,z)
        position_uniform_min = [0.4, -0.1, 0.045]  # position (x,y,z)
        position_uniform_max = [0.4, 0.1, 0.045]  # position (x,y,z)

    @configclass
    class GoalPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.2, 0.0]  # position default (x,y,z)
        position_uniform_min = [0.4, -0.2, 0.]  # position (x,y,z)
        position_uniform_max = [0.5, 0.2, 0.]  # position (x,y,z)

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()


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
        # tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        # tool_orientations = {"scale": 1.0}
        # -- object state
        object_positions = {"scale": 1.0}
        object_orientations = {"scale": 1.0}
        object_relative_tool_positions = {"scale": 1.0}
        # object_relative_tool_orientations = {"scale": 1.0}
        # -- object desired state
        object_desired_positions = {"scale": 1.0}
        object_to_goal_positions = {"scale": 1.0}
        # -- previous action
        # arm_actions = {"scale": 1.0}
        # tool_actions = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- robot-centric
    # reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.15}
    tracking_object_position_tanh = {"weight": 5.0, "sigma": 0.2, "threshold": 0.11}
    # reaching_object_position_exp = {"weight": 2.5, "sigma": 0.1}
    # tracking_object_position_exp = {"weight": 5.0, "sigma": 0.2, "threshold": 0.1}
    # push_object_success = {"weight": 3.5, "threshold": 0.04}
    reaching_object_position_negative = {"weight": 1,}
    # tracking_object_position_negative = {"weight": 3,}
    push_object_success = {"weight": 3.5, "threshold": 0.05}


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
    # control_type = "default"  # "default", "inverse_kinematics"
    # control_type = "inverse_kinematics"  # "default", "inverse_kinematics"
    control_type = "differential_inverse_kinematics"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    # inverse_kinematics: InverseKinematicsCfg = InverseKinematicsCfg(
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="position_rel",
        ik_method="dls",
        position_command_scale=(0.05, 0.05, 0.05),
        rotation_command_scale=(0.05, 0.05, 0.05),
    )


##
# Environment configuration
##


@configclass
class PushEnvCfg(IsaacEnvCfg):
    """Configuration for the Push environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=4.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=False, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=0.01,
        substeps=1,
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
    FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG.init_state.dof_pos['panda_finger_joint*'] = 0
    robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # -- goal
    goal: GoalMarkerCfg = GoalMarkerCfg()
    # -- table
    table: TableCfg = TableCfg()
    # -- visualization marker
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

