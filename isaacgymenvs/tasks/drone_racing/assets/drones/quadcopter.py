import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import urdfpy
from scipy.spatial.transform import Rotation

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from ..utils.urdf_utils import export_urdf


@dataclass
class DroneQuadcopterOptions:
    """
    Configuration for the quadcopter in FLU body frame convention.

    The center of body frame is the crossing point of the arms,
    on the upper surface of the arm plate.

    Collision shape is the minimum bounding box of the quadcopter,
    and is automatically computed.
    """

    # file name
    file_name: str = "drone_quadcopter"

    # length of the two front arms [m]
    arm_length_front: float = 0.125

    # length of the two back arms [m]
    arm_length_back: float = 0.125

    # thickness of the arm plate [m]
    arm_thickness: float = 0.01

    # separation angle between two front arms [rad]
    arm_front_angle: float = 1.780236

    # diameter of the motor cylinder [m]
    motor_diameter: float = 0.023

    # height of the motor cylinder [m]
    motor_height: float = 0.006

    # central body cuboid position in body frame [m]
    central_body_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.015])

    # central body cuboid dimension [m]
    central_body_dim: List[float] = field(default_factory=lambda: [0.15, 0.05, 0.05])

    # propeller cylinder diameter [m]
    propeller_diameter: float = 0.12954

    # propeller cylinder height [m]
    propeller_height: float = 0.01

    # mass of the whole quadcopter treated as a rigid body
    mass: float = 0.752

    # center of mass position in body frame [m]
    center_of_mass: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # diagonal inertia in [kg m^2]
    diagonal_inertia: List[float] = field(
        default_factory=lambda: [0.0025, 0.0021, 0.0043]
    )

    # quaternion representing the principle axes matrix [w, x, y, z]
    principle_axes_q: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    # options for importing into Isaac Gym
    asset_options: AssetOptions = AssetOptions()

    # disable visuals to avoid blocking camera sensors
    disable_visuals: bool = False


def create_drone_quadcopter(
    gym: Gym, sim: Sim, options: DroneQuadcopterOptions
) -> Asset:
    """
    Create a quadcopter asset.

    Args:
        gym: returned by ``acquire_gym``.
        sim: simulation handle.
        options: options for visual, collision, inertia, and importing.

    Returns:
        - An asset object as the return of calling ``load_asset``.
    """

    # ========== inertial ==========

    principle_axes_r = Rotation.from_quat(
        np.array(options.principle_axes_q)[[1, 2, 3, 0]]
    )
    inertial_origin = np.zeros((4, 4))
    inertial_origin[:3, :3] = principle_axes_r.as_matrix()
    inertial_origin[:3, -1] = options.center_of_mass
    inertial_origin[-1, -1] = 1
    inertial = urdfpy.Inertial(
        mass=options.mass,
        inertia=np.diag(options.diagonal_inertia),
        origin=inertial_origin,
    )

    # ========== collisions ==========

    collisions: List[urdfpy.Collision] = []

    collision_box_pos, collision_box_dim = _get_collision_box(options)
    collision_geometry = urdfpy.Geometry(box=urdfpy.Box(collision_box_dim))
    collision_origin = urdfpy.xyz_rpy_to_matrix(collision_box_pos + [0, 0, 0])
    collision = urdfpy.Collision(
        name=None, origin=collision_origin, geometry=collision_geometry
    )
    collisions.append(collision)

    # ========== visuals ==========

    visuals: List[urdfpy.Visual] = []

    # central body
    central_body_geometry = urdfpy.Geometry(box=urdfpy.Box(options.central_body_dim))
    central_body_origin = urdfpy.xyz_rpy_to_matrix(options.central_body_pos + [0, 0, 0])
    central_body_visual = urdfpy.Visual(
        geometry=central_body_geometry, origin=central_body_origin
    )
    visuals.append(central_body_visual)

    # arm 1, 4
    arm_offset = (
        options.arm_length_front + options.arm_length_back
    ) / 2 - options.arm_length_back
    arm_14_geometry = urdfpy.Geometry(
        box=urdfpy.Box(
            [
                options.arm_length_front + options.arm_length_back,
                options.motor_diameter,
                options.arm_thickness,
            ]
        )
    )
    arm_14_xyz = [
        math.cos(options.arm_front_angle / 2) * arm_offset,
        math.sin(options.arm_front_angle / 2) * arm_offset,
        -options.arm_thickness / 2,
    ]

    arm_14_rpy = [0, 0, options.arm_front_angle / 2]
    arm_14_origin = urdfpy.xyz_rpy_to_matrix(arm_14_xyz + arm_14_rpy)
    arm_14_visual = urdfpy.Visual(geometry=arm_14_geometry, origin=arm_14_origin)
    visuals.append(arm_14_visual)

    # arm 2, 3
    arm_23_geometry = arm_14_geometry
    arm_23_xyz = [
        math.cos(options.arm_front_angle / 2) * arm_offset,
        -math.sin(options.arm_front_angle / 2) * arm_offset,
        -options.arm_thickness / 2,
    ]

    arm_23_rpy = [0, 0, -options.arm_front_angle / 2]
    arm_23_origin = urdfpy.xyz_rpy_to_matrix(arm_23_xyz + arm_23_rpy)
    arm_23_visual = urdfpy.Visual(geometry=arm_23_geometry, origin=arm_23_origin)
    visuals.append(arm_23_visual)

    # rotors
    rotor_angles = [
        options.arm_front_angle / 2 + math.pi,
        -options.arm_front_angle / 2,
        -options.arm_front_angle / 2 + math.pi,
        options.arm_front_angle / 2,
    ]
    for i in [1, 2, 3, 4]:
        if i == 1 or i == 3:
            arm_length = options.arm_length_back
        else:
            arm_length = options.arm_length_front
        # motor
        motor_geometry = urdfpy.Geometry(
            cylinder=urdfpy.Cylinder(
                radius=options.motor_diameter / 2,
                length=options.motor_height + options.arm_thickness,
            )
        )
        motor_xyz = [
            math.cos(rotor_angles[i - 1]) * arm_length,
            math.sin(rotor_angles[i - 1]) * arm_length,
            (options.motor_height + options.arm_thickness) / 2 - options.arm_thickness,
        ]

        motor_origin = urdfpy.xyz_rpy_to_matrix(motor_xyz + [0, 0, 0])
        motor_visual = urdfpy.Visual(geometry=motor_geometry, origin=motor_origin)
        visuals.append(motor_visual)
        # propeller
        propeller_geometry = urdfpy.Geometry(
            cylinder=urdfpy.Cylinder(
                radius=options.propeller_diameter / 2,
                length=options.propeller_height,
            )
        )
        propeller_xyz = [
            math.cos(rotor_angles[i - 1]) * arm_length,
            math.sin(rotor_angles[i - 1]) * arm_length,
            options.propeller_height / 2 + options.motor_height,
        ]

        propeller_origin = urdfpy.xyz_rpy_to_matrix(propeller_xyz + [0, 0, 0])
        propeller_material = None
        if i == 2 or i == 4:
            propeller_material = urdfpy.Material("red", color=[1.0, 0.0, 0.0, 1.0])
        propeller_visual = urdfpy.Visual(
            geometry=propeller_geometry,
            origin=propeller_origin,
            material=propeller_material,
        )
        visuals.append(propeller_visual)

    # ========== quad ==========

    if options.disable_visuals:
        visuals = []
    link = urdfpy.Link(
        name="base", inertial=inertial, visuals=visuals, collisions=collisions
    )
    links: List[urdfpy.Link] = [link]

    quad = urdfpy.URDF(name=options.file_name, links=links)

    # ========== create file ==========

    file_dir, file_name_ext = export_urdf(quad)

    # ========== load asset ==========

    asset = gym.load_asset(sim, file_dir, file_name_ext, options.asset_options)
    return asset


def _get_collision_box(
    options: DroneQuadcopterOptions,
) -> Tuple[List[float], List[float]]:
    positive_x_extend = max(
        options.central_body_pos[0] + options.central_body_dim[0] / 2,
        options.arm_length_front * math.cos(options.arm_front_angle / 2)
        + options.propeller_diameter / 2,
    )
    negative_x_extend = -min(
        options.central_body_pos[0] - options.central_body_dim[0] / 2,
        options.arm_length_back * math.cos(options.arm_front_angle / 2 + math.pi)
        - options.propeller_diameter / 2,
    )
    collision_bbox_length_x = positive_x_extend + negative_x_extend
    collision_bbox_center_x = -negative_x_extend + collision_bbox_length_x / 2

    positive_y_extend = max(
        options.central_body_pos[1] + options.central_body_dim[1] / 2,
        options.arm_length_front * math.sin(options.arm_front_angle / 2)
        + options.propeller_diameter / 2,
        options.arm_length_back * math.sin(math.pi - options.arm_front_angle / 2)
        + options.propeller_diameter / 2,
    )
    collision_bbox_length_y = 2 * positive_y_extend
    collision_bbox_center_y = 0.0

    positive_z_extend = max(
        options.central_body_pos[2] + options.central_body_dim[2] / 2,
        options.motor_height + options.propeller_height,
    )
    negative_z_extend = -min(
        options.central_body_pos[2] - options.central_body_dim[2] / 2,
        -options.arm_thickness,
    )
    collision_bbox_length_z = positive_z_extend + negative_z_extend
    collision_bbox_center_z = -negative_z_extend + collision_bbox_length_z / 2

    position = [
        collision_bbox_center_x,
        collision_bbox_center_y,
        collision_bbox_center_z,
    ]
    dimension = [
        collision_bbox_length_x,
        collision_bbox_length_y,
        collision_bbox_length_z,
    ]

    return position, dimension
