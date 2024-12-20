import math
import os
from typing import List, Tuple

import torch
import urdfpy


def cuboid_link(name: str, size: List[float], color: List[float] = None) -> urdfpy.Link:
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]
    geometry = urdfpy.Geometry(box=urdfpy.Box(size))
    link = urdfpy.Link(
        name=name,
        inertial=None,
        visuals=[urdfpy.Visual(geometry=geometry)],
        collisions=[urdfpy.Collision(name=None, origin=None, geometry=geometry)],
    )
    set_link_color(link, color)

    return link


def cylinder_link(
    name: str,
    radius: float,
    length: float,
    align_x: bool = False,
    offset: List[float] = None,
    color: List[float] = None,
) -> urdfpy.Link:
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]
    geometry = urdfpy.Geometry(cylinder=urdfpy.Cylinder(radius=radius, length=length))
    if align_x:
        origin = urdfpy.xyz_rpy_to_matrix(offset + [0.0, math.radians(90), 0.0])
    else:
        origin = urdfpy.xyz_rpy_to_matrix(offset + [0.0, 0.0, 0.0])
    link = urdfpy.Link(
        name=name,
        inertial=None,
        visuals=[urdfpy.Visual(geometry=geometry, origin=origin)],
        collisions=[urdfpy.Collision(name=None, origin=origin, geometry=geometry)],
    )
    set_link_color(link, color)

    return link


def sphere_link(name: str, radius: float, color: List[float] = None) -> urdfpy.Link:
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]
    geometry = urdfpy.Geometry(sphere=urdfpy.Sphere(radius=radius))
    link = urdfpy.Link(
        name=name,
        inertial=None,
        visuals=[urdfpy.Visual(geometry=geometry)],
        collisions=[urdfpy.Collision(name=None, origin=None, geometry=geometry)],
    )
    set_link_color(link, color)

    return link


def hollow_cuboid_link(
    name: str,
    length_x: float,
    inner_length_y: float,
    outer_length_y: float,
    inner_length_z: float,
    outer_length_z: float,
    color: List[float] = None,
) -> urdfpy.Link:
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]

    # geometry
    horizontal_bar_geometry = urdfpy.Geometry(
        box=urdfpy.Box(
            [length_x, outer_length_y, (outer_length_z - inner_length_z) / 2]
        )
    )
    vertical_bar_geometry = urdfpy.Geometry(
        box=urdfpy.Box(
            [length_x, (outer_length_y - inner_length_y) / 2, outer_length_z]
        )
    )

    # origin
    top_bar_origin = urdfpy.xyz_rpy_to_matrix(
        [0, 0, (outer_length_z + inner_length_z) / 4, 0, 0, 0]
    )
    bottom_bar_origin = urdfpy.xyz_rpy_to_matrix(
        [0, 0, -(outer_length_z + inner_length_z) / 4, 0, 0, 0]
    )
    left_bar_origin = urdfpy.xyz_rpy_to_matrix(
        [0, (outer_length_y + inner_length_y) / 4, 0, 0, 0, 0]
    )
    right_bar_origin = urdfpy.xyz_rpy_to_matrix(
        [0, -(outer_length_y + inner_length_y) / 4, 0, 0, 0, 0]
    )

    # visual
    top_bar_visual = urdfpy.Visual(
        geometry=horizontal_bar_geometry, origin=top_bar_origin
    )
    bottom_bar_visual = urdfpy.Visual(
        geometry=horizontal_bar_geometry, origin=bottom_bar_origin
    )
    left_bar_visual = urdfpy.Visual(
        geometry=vertical_bar_geometry, origin=left_bar_origin
    )
    right_bar_visual = urdfpy.Visual(
        geometry=vertical_bar_geometry, origin=right_bar_origin
    )

    # collision
    top_bar_collision = urdfpy.Collision(
        name=None, geometry=horizontal_bar_geometry, origin=top_bar_origin
    )
    bottom_bar_collision = urdfpy.Collision(
        name=None, geometry=horizontal_bar_geometry, origin=bottom_bar_origin
    )
    left_bar_collision = urdfpy.Collision(
        name=None, geometry=vertical_bar_geometry, origin=left_bar_origin
    )
    right_bar_collision = urdfpy.Collision(
        name=None, geometry=vertical_bar_geometry, origin=right_bar_origin
    )

    # link
    link = urdfpy.Link(
        name=name,
        inertial=None,
        visuals=[top_bar_visual, bottom_bar_visual, left_bar_visual, right_bar_visual],
        collisions=[
            top_bar_collision,
            bottom_bar_collision,
            left_bar_collision,
            right_bar_collision,
        ],
    )
    set_link_color(link, color)

    return link


def cuboid_wireframe_link(
    name: str,
    size: List[float],
    weight: float,
    color: List[float] = None,
) -> urdfpy.Link:
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]

    # geometry
    geometry_x = urdfpy.Geometry(box=urdfpy.Box([size[0] + weight, weight, weight]))
    geometry_y = urdfpy.Geometry(box=urdfpy.Box([weight, size[1] + weight, weight]))
    geometry_z = urdfpy.Geometry(box=urdfpy.Box([weight, weight, size[2] + weight]))

    # origin
    x_bar_upper_left_origin = urdfpy.xyz_rpy_to_matrix(
        [0, size[1] / 2, size[2] / 2, 0, 0, 0]
    )
    x_bar_lower_left_origin = urdfpy.xyz_rpy_to_matrix(
        [0, size[1] / 2, -size[2] / 2, 0, 0, 0]
    )
    x_bar_upper_right_origin = urdfpy.xyz_rpy_to_matrix(
        [0, -size[1] / 2, size[2] / 2, 0, 0, 0]
    )
    x_bar_lower_right_origin = urdfpy.xyz_rpy_to_matrix(
        [0, -size[1] / 2, -size[2] / 2, 0, 0, 0]
    )

    y_bar_upper_front_origin = urdfpy.xyz_rpy_to_matrix(
        [size[0] / 2, 0, size[2] / 2, 0, 0, 0]
    )
    y_bar_lower_front_origin = urdfpy.xyz_rpy_to_matrix(
        [size[0] / 2, 0, -size[2] / 2, 0, 0, 0]
    )
    y_bar_upper_back_origin = urdfpy.xyz_rpy_to_matrix(
        [-size[0] / 2, 0, size[2] / 2, 0, 0, 0]
    )
    y_bar_lower_back_origin = urdfpy.xyz_rpy_to_matrix(
        [-size[0] / 2, 0, -size[2] / 2, 0, 0, 0]
    )

    z_bar_front_left_origin = urdfpy.xyz_rpy_to_matrix(
        [size[0] / 2, size[1] / 2, 0, 0, 0, 0]
    )
    z_bar_front_right_origin = urdfpy.xyz_rpy_to_matrix(
        [size[0] / 2, -size[1] / 2, 0, 0, 0, 0]
    )
    z_bar_back_left_origin = urdfpy.xyz_rpy_to_matrix(
        [-size[0] / 2, size[1] / 2, 0, 0, 0, 0]
    )
    z_bar_back_right_origin = urdfpy.xyz_rpy_to_matrix(
        [-size[0] / 2, -size[1] / 2, 0, 0, 0, 0]
    )

    # visual
    x_bar_upper_left_visual = urdfpy.Visual(
        geometry=geometry_x, origin=x_bar_upper_left_origin
    )
    x_bar_lower_left_visual = urdfpy.Visual(
        geometry=geometry_x, origin=x_bar_lower_left_origin
    )
    x_bar_upper_right_visual = urdfpy.Visual(
        geometry=geometry_x, origin=x_bar_upper_right_origin
    )
    x_bar_lower_right_visual = urdfpy.Visual(
        geometry=geometry_x, origin=x_bar_lower_right_origin
    )

    y_bar_upper_front_visual = urdfpy.Visual(
        geometry=geometry_y, origin=y_bar_upper_front_origin
    )
    y_bar_lower_front_visual = urdfpy.Visual(
        geometry=geometry_y, origin=y_bar_lower_front_origin
    )
    y_bar_upper_back_visual = urdfpy.Visual(
        geometry=geometry_y, origin=y_bar_upper_back_origin
    )
    y_bar_lower_back_visual = urdfpy.Visual(
        geometry=geometry_y, origin=y_bar_lower_back_origin
    )

    z_bar_front_left_visual = urdfpy.Visual(
        geometry=geometry_z, origin=z_bar_front_left_origin
    )
    z_bar_front_right_visual = urdfpy.Visual(
        geometry=geometry_z, origin=z_bar_front_right_origin
    )
    z_bar_back_left_visual = urdfpy.Visual(
        geometry=geometry_z, origin=z_bar_back_left_origin
    )
    z_bar_back_right_visual = urdfpy.Visual(
        geometry=geometry_z, origin=z_bar_back_right_origin
    )

    # collision
    x_bar_upper_left_collision = urdfpy.Collision(
        name=None, geometry=geometry_x, origin=x_bar_upper_left_origin
    )
    x_bar_lower_left_collision = urdfpy.Collision(
        name=None, geometry=geometry_x, origin=x_bar_lower_left_origin
    )
    x_bar_upper_right_collision = urdfpy.Collision(
        name=None, geometry=geometry_x, origin=x_bar_upper_right_origin
    )
    x_bar_lower_right_collision = urdfpy.Collision(
        name=None, geometry=geometry_x, origin=x_bar_lower_right_origin
    )

    y_bar_upper_front_collision = urdfpy.Collision(
        name=None, geometry=geometry_y, origin=y_bar_upper_front_origin
    )
    y_bar_lower_front_collision = urdfpy.Collision(
        name=None, geometry=geometry_y, origin=y_bar_lower_front_origin
    )
    y_bar_upper_back_collision = urdfpy.Collision(
        name=None, geometry=geometry_y, origin=y_bar_upper_back_origin
    )
    y_bar_lower_back_collision = urdfpy.Collision(
        name=None, geometry=geometry_y, origin=y_bar_lower_back_origin
    )

    z_bar_front_left_collision = urdfpy.Collision(
        name=None, geometry=geometry_z, origin=z_bar_front_left_origin
    )
    z_bar_front_right_collision = urdfpy.Collision(
        name=None, geometry=geometry_z, origin=z_bar_front_right_origin
    )
    z_bar_back_left_collision = urdfpy.Collision(
        name=None, geometry=geometry_z, origin=z_bar_back_left_origin
    )
    z_bar_back_right_collision = urdfpy.Collision(
        name=None, geometry=geometry_z, origin=z_bar_back_right_origin
    )

    # link
    link = urdfpy.Link(
        name=name,
        inertial=None,
        visuals=[
            x_bar_upper_left_visual,
            x_bar_lower_left_visual,
            x_bar_upper_right_visual,
            x_bar_lower_right_visual,
            y_bar_upper_front_visual,
            y_bar_lower_front_visual,
            y_bar_upper_back_visual,
            y_bar_lower_back_visual,
            z_bar_front_left_visual,
            z_bar_front_right_visual,
            z_bar_back_left_visual,
            z_bar_back_right_visual,
        ],
        collisions=[
            x_bar_upper_left_collision,
            x_bar_lower_left_collision,
            x_bar_upper_right_collision,
            x_bar_lower_right_collision,
            y_bar_upper_front_collision,
            y_bar_lower_front_collision,
            y_bar_upper_back_collision,
            y_bar_lower_back_collision,
            z_bar_front_left_collision,
            z_bar_front_right_collision,
            z_bar_back_left_collision,
            z_bar_back_right_collision,
        ],
    )
    set_link_color(link, color)

    return link


def random_geometries_link(
    name: str,
    num_geometries: int,
    space_dim: List[float],
    space_offset: List[float],
    min_geometry_size: float,
    max_geometry_size: float,
    color: List[float] = None,
) -> urdfpy.Link:
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]

    # generate a random tensor
    # geometry type (1) + xyz_rpy (6) + size (3)
    # for box the size represent edge lengths
    # for cylinder the size represent diameter and length, 1 value unused
    # for sphere only the first value represent diameter
    random_tensor = torch.rand(num_geometries, 1 + 6 + 3)
    random_tensor[:, 0] //= 1 / 3
    random_tensor[:, 1] = (
        random_tensor[:, 1] * space_dim[0] - space_dim[0] / 2 + space_offset[0]
    )
    random_tensor[:, 2] = (
        random_tensor[:, 2] * space_dim[1] - space_dim[1] / 2 + space_offset[1]
    )
    random_tensor[:, 3] = (
        random_tensor[:, 3] * space_dim[2] - space_dim[2] / 2 + space_offset[2]
    )
    random_tensor[:, 4:7] *= torch.pi * 2
    range_geometry_size = max_geometry_size - min_geometry_size
    random_tensor[:, 7:] = (
        random_tensor[:, 7:] * range_geometry_size + min_geometry_size
    )

    visuals: List[urdfpy.Visual] = []
    collisions: List[urdfpy.Collision] = []
    for i in range(num_geometries):
        if int(random_tensor[i, 0]) == 0:
            # box
            geometry = urdfpy.Geometry(box=urdfpy.Box(random_tensor[i, 7:].tolist()))
        elif int(random_tensor[i, 0]) == 1:
            # cylinder
            r = float(random_tensor[i, 7]) / 2
            length = float(random_tensor[i, 8])
            geometry = urdfpy.Geometry(
                cylinder=urdfpy.Cylinder(radius=r, length=length)
            )
        else:
            # sphere
            r = float(random_tensor[i, 7]) / 2
            geometry = urdfpy.Geometry(sphere=urdfpy.Sphere(r))
        origin = urdfpy.xyz_rpy_to_matrix(random_tensor[i, 1:7].tolist())
        visual = urdfpy.Visual(geometry=geometry, origin=origin)
        collision = urdfpy.Collision(name=None, geometry=geometry, origin=origin)
        visuals.append(visual)
        collisions.append(collision)

    link = urdfpy.Link(name=name, inertial=None, visuals=visuals, collisions=collisions)
    set_link_color(link, color)

    return link


def fixed_joint(
    parent_name: str, child_name: str, xyz_rpy: List[float]
) -> urdfpy.Joint:
    joint = urdfpy.Joint(
        name=parent_name + "_" + child_name,
        parent=parent_name,
        child=child_name,
        joint_type="fixed",
        origin=urdfpy.xyz_rpy_to_matrix(xyz_rpy),
    )
    return joint


def set_link_color(link: urdfpy.Link, color: List[float]):
    visuals: List[urdfpy.Visual] = link.visuals
    for visual in visuals:
        visual.material = urdfpy.Material("color" + str(color), color=color)


def export_urdf(urdf: urdfpy.URDF) -> Tuple[str, str]:
    file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name_with_ext = urdf.name + ".urdf"
    file = os.path.join(file_dir, file_name_with_ext)
    urdf.save(file)

    return file_dir, file_name_with_ext
