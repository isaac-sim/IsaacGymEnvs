import argparse
import time
from math import sin, cos, radians

import numpy as np
import plotly.graph_objects as go

from isaacgym import gymapi
from isaacgymenvs.tasks.drone_racing.mdp import RewardParams, Reward
from isaacgymenvs.tasks.drone_racing.waypoint import WaypointData

print("Importing torch...")
import torch  # noqa


def pt_to_px(pt):
    return pt * (96 / 72)


if __name__ == "__main__":
    # info
    print("+++ Plotting guidance reward")

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--wp_w", type=float, default=3)
    arg_parser.add_argument("--wp_h", type=float, default=1.6)
    arg_parser.add_argument("--space_x", type=float, default=4)
    arg_parser.add_argument("--space_y", type=float, default=4)
    arg_parser.add_argument("--space_z", type=float, default=4)
    arg_parser.add_argument("--points_x", type=int, default=21)
    arg_parser.add_argument("--points_y", type=int, default=100)
    arg_parser.add_argument("--points_z", type=int, default=100)
    arg_parser.add_argument("--fig_x_scale", type=float, default=7.5)
    arg_parser.add_argument("--fig_w", type=float, required=True)
    arg_parser.add_argument("--fig_h", type=float, required=True)
    arg_parser.add_argument("--fig_file", type=str, required=True)
    arg_parser.add_argument("--font_size", type=float, default=7)
    arg_parser.add_argument("--cam_dist", type=float, default=2)
    arg_parser.add_argument("--cam_azm", type=float, default=65)
    arg_parser.add_argument("--cam_ele", type=float, default=15)
    args = arg_parser.parse_args()

    # settings
    compute_device = args.device
    wp_w = args.wp_w
    wp_h = args.wp_h
    space_len_x = args.space_x
    space_len_y = args.space_y
    space_len_z = args.space_z
    num_points_x = args.points_x
    num_points_y = args.points_y
    num_points_z = args.points_z
    fig_x_axis_scale = args.fig_x_scale
    font_size = args.font_size
    cam_r_to_center = args.cam_dist
    cam_angles = [args.cam_azm, args.cam_ele]
    w_pt = args.fig_w
    h_pt = args.fig_h
    fig_f = args.fig_file

    # generate the grid points
    x_range = torch.linspace(-space_len_x, space_len_x, num_points_x)
    y_range = torch.linspace(-space_len_y, space_len_y, num_points_y)
    z_range = torch.linspace(-space_len_z, space_len_z, num_points_z)

    # create a meshgrid using torch
    x_grid, y_grid, z_grid = torch.meshgrid(x_range, y_range, z_range, indexing="ij")

    # flatten the grid points
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = z_grid.flatten()
    points = torch.stack((x, y, z), dim=-1)
    num_points = points.shape[0]

    # use parallel envs to compute the reward
    num_envs = num_points

    # create waypoint data
    wp_quaternion = torch.zeros(num_envs, 2, 4)
    wp_quaternion[:, :, -1] = 1
    wp_data = WaypointData(
        position=torch.zeros(num_envs, 2, 3),
        quaternion=wp_quaternion,
        width=torch.ones(num_envs, 2) * wp_w,
        height=torch.ones(num_envs, 2) * wp_h,
        gate_flag=torch.zeros(num_envs, 2, dtype=torch.bool),
        gate_x_len_choice=torch.zeros(num_envs, 2),
        gate_weight_choice=torch.zeros(num_envs, 2),
        psi=torch.zeros(num_envs, 1),
        theta=torch.zeros(num_envs, 1),
        gamma=torch.zeros(num_envs, 1),
        r=torch.zeros(num_envs, 1),
    )

    # create reward calculator
    reward_params = RewardParams()
    reward_params.num_envs = num_envs
    reward_params.device = compute_device
    reward_params.guidance_x_thresh = 3
    # reward_params.k_guidance = 0.1
    # reward_params.k_rejection = 10
    mdp_reward = Reward(reward_params)

    # compute reward and get guidance term
    drone_state = torch.zeros(wp_data.num_envs, 13, device=compute_device)
    drone_state[:, 6] = 1.0
    action = torch.zeros(wp_data.num_envs, 4, device=compute_device)
    mdp_reward.set_waypoint_and_cam(
        wp_data=wp_data,
        cam_tf=[gymapi.Transform()] * wp_data.num_envs,
    )
    mdp_reward.set_init_drone_state_action(drone_state, action)

    drone_state[:, :3] = points.to(device=compute_device)
    drone_collision = timeout = wp_passing = torch.zeros(
        wp_data.num_envs, dtype=torch.bool, device=compute_device
    )
    next_wp_id = torch.ones(wp_data.num_envs, dtype=torch.int, device=compute_device)
    t0 = time.time()
    r = mdp_reward.compute(
        drone_state, action, drone_collision, timeout, wp_passing, next_wp_id
    )
    r_guidance = mdp_reward.reward_guidance
    t1 = time.time()
    print("number of points:", num_points)
    print("calculation:", int((t1 - t0) * 1000), "ms")

    # reward field
    reward_field = go.Scatter3d(
        x=x.numpy(),
        y=y.numpy(),
        z=z.numpy(),
        mode="markers",
        marker=dict(
            size=0.5,
            color=r_guidance.cpu().numpy(),  # Set color to the rewards
            colorscale="Viridis",  # Choose a colorscale
            colorbar=dict(
                title="",
                orientation="h",
                thickness=6,
                tickfont=dict(size=pt_to_px(font_size / 1.25)),
            ),  # Show colorbar
            opacity=0.3,
        ),
    )

    # gate
    x_gate = [0, 0, 0, 0, 0]  # Closed loop
    y_gate = [-wp_w / 2, -wp_w / 2, wp_w / 2, wp_w / 2, -wp_w / 2]
    z_gate = [-wp_h / 2, wp_h / 2, wp_h / 2, -wp_h / 2, -wp_h / 2]
    gate = go.Scatter3d(
        x=x_gate,
        y=y_gate,
        z=z_gate,
        mode="lines",  # Use lines to create the outline
        line=dict(color="black", width=2),  # Outline color and width
    )

    # frame
    x_axis = go.Scatter3d(
        x=[0, 1 / fig_x_axis_scale],
        y=[0, 0],
        z=[0, 0],
        mode="lines",
        line=dict(color="red", width=2),
    )
    y_axis = go.Scatter3d(
        x=[0, 0],
        y=[0, 1],
        z=[0, 0],
        mode="lines",
        line=dict(color="green", width=2),
    )
    z_axis = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 1],
        mode="lines",
        line=dict(color="blue", width=2),
    )

    layout = go.Layout(
        # title="Guidance Reward Field",
        width=pt_to_px(w_pt),
        height=pt_to_px(h_pt),
        scene=dict(
            xaxis=dict(
                title="",
                tickvals=np.arange(-space_len_x, space_len_x + 0.4, 0.4),
                ticktext=[
                    f"{val:.1f}"
                    for val in np.arange(-space_len_x, space_len_x + 0.4, 0.4)
                ],
                tickfont=dict(size=pt_to_px(font_size)),  # Set smaller tick font size
            ),
            yaxis=dict(title="", tickvals=[], ticktext=[]),
            zaxis=dict(title="", tickvals=[], ticktext=[]),
            camera=dict(
                eye=dict(
                    x=cam_r_to_center
                    * cos(radians(cam_angles[1]))
                    * cos(radians(cam_angles[0])),
                    y=cam_r_to_center
                    * cos(radians(cam_angles[1]))
                    * sin(radians(cam_angles[0])),
                    z=cam_r_to_center * sin(radians(cam_angles[1])),
                )
            ),
            aspectmode="manual",
            aspectratio=dict(x=fig_x_axis_scale, y=1, z=1),
        ),
        showlegend=False,
        font=dict(family="Times New Roman"),
        margin=dict(l=2, r=2, t=2, b=2, pad=0),
    )
    fig = go.Figure(data=[reward_field, gate, x_axis, y_axis, z_axis], layout=layout)
    fig.write_image(fig_f, format="pdf", engine="kaleido", scale=10)
