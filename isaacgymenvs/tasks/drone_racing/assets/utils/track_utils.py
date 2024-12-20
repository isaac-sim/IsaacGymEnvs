import math
from typing import List, Tuple

import urdfpy

from isaacgym.gymapi import Gym, Sim, AssetOptions, Asset
from .track_options import TrackOptions
from .urdf_utils import (
    hollow_cuboid_link,
    cylinder_link,
    sphere_link,
    fixed_joint,
    cuboid_link,
    set_link_color,
    export_urdf,
)
from ...waypoint import Waypoint


def create_track_asset(
    name: str,
    track_options: TrackOptions,
    waypoints: List[Waypoint],
    obstacle_links: List[urdfpy.Link],
    obstacle_origins: List[List[float]],
    obstacle_flags: List[bool],
    asset_options: AssetOptions,
    gym: Gym,
    sim: Sim,
) -> Asset:
    # urdf
    track_urdf = create_track_urdf(
        name, track_options, waypoints, obstacle_links, obstacle_origins, obstacle_flags
    )

    # file
    file_dir, file_name_ext = export_urdf(track_urdf)

    # asset
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = True
    asset = gym.load_asset(sim, file_dir, file_name_ext, asset_options)

    return asset


def create_track_urdf(
    name: str,
    options: TrackOptions,
    waypoints: List[Waypoint],
    obstacle_links: List[urdfpy.Link],
    obstacle_origins: List[List[float]],
    obstacle_flags: List[bool],
) -> urdfpy.URDF:
    links: List[urdfpy.Link] = []
    joints: List[urdfpy.Joint] = []

    # dummy base link
    links.append(urdfpy.Link("base", None, None, None))

    # obstacle links and joints
    assert len(obstacle_links) == len(obstacle_origins) == len(obstacle_flags)
    num_obstacles = len(obstacle_links)
    for i in range(num_obstacles):
        if obstacle_flags[i]:
            set_link_color(obstacle_links[i], options.additional_obstacle_color)
        links.append(obstacle_links[i])
        joints.append(fixed_joint("base", obstacle_links[i].name, obstacle_origins[i]))

    # gate links and joints
    for waypoint in waypoints:
        if waypoint.gate:
            gate_link = hollow_cuboid_link(
                name="gate_" + str(waypoint.index),
                length_x=options.gate_length_x,
                inner_length_y=waypoint.length_y,
                outer_length_y=waypoint.length_y + options.gate_size,
                inner_length_z=waypoint.length_z,
                outer_length_z=waypoint.length_z + options.gate_size,
                color=options.gate_color,
            )
            xyz_rpy = waypoint.xyz + waypoint.rpy_rad()
            links.append(gate_link)
            joints.append(fixed_joint("base", gate_link.name, xyz_rpy))

    # debug links and joints
    if options.enable_debug_visualization:
        num_waypoints = len(waypoints)
        for i in range(num_waypoints):
            # line segments
            if not i == num_waypoints - 1:
                origin_xyz_rpy, length = get_line_segment(
                    waypoints[i].xyz, waypoints[i + 1].xyz
                )
                line_link = cylinder_link(
                    "line_" + str(i),
                    options.debug_cylinder_radius,
                    length,
                    True,
                    [0.0, 0.0, 0.0],
                    options.debug_visual_color,
                )
                links.append(line_link)
                joints.append(fixed_joint("base", line_link.name, origin_xyz_rpy))

            # waypoint centers
            center_link = sphere_link(
                "center_" + str(i),
                options.debug_cylinder_radius,
                options.debug_visual_color,
            )
            links.append(center_link)
            joints.append(
                fixed_joint("base", center_link.name, waypoints[i].xyz + [0, 0, 0])
            )

            # waypoint directions
            direction_link = cylinder_link(
                "direction_" + str(i),
                options.debug_cylinder_radius,
                0.3,
                True,
                [0.15, 0.0, 0.0],
                options.gate_color,
            )
            links.append(direction_link)
            joints.append(
                fixed_joint(
                    "base",
                    direction_link.name,
                    waypoints[i].xyz + waypoints[i].rpy_rad(),
                )
            )

            # waypoint cuboids
            region_link = cuboid_link(
                "region_" + str(i),
                [
                    options.debug_waypoint_length_x,
                    waypoints[i].length_y,
                    waypoints[i].length_z,
                ],
                options.debug_visual_color,
            )
            links.append(region_link)
            joints.append(
                fixed_joint(
                    "base", region_link.name, waypoints[i].xyz + waypoints[i].rpy_rad()
                )
            )

    urdf = urdfpy.URDF(name=name, links=links, joints=joints)
    return urdf


def get_line_segment(
    xyz_a: List[float], xyz_b: List[float]
) -> Tuple[List[float], float]:
    x_a, y_a, z_a = xyz_a
    x_b, y_b, z_b = xyz_b
    x_d = x_b - x_a
    y_d = y_b - y_a
    z_d = z_b - z_a
    dist = (x_d**2 + y_d**2 + z_d**2) ** 0.5
    yaw = math.atan2(y_d, x_d)
    pitch = -math.atan2(z_d, (x_d**2 + y_d**2) ** 0.5)
    xyz_rpy = [0.5 * (x_a + x_b), 0.5 * (y_a + y_b), 0.5 * (z_a + z_b), 0.0, pitch, yaw]
    return xyz_rpy, dist
