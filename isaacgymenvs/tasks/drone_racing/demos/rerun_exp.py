import argparse
import multiprocessing as mp
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List

import matplotlib
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from rerun_loader_urdf import URDFLogger


# TODO: blueprint
def rrb_single_env() -> rrb.Blueprint:
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    return blueprint


# TODO: blueprint
def rrb_combined_env() -> rrb.Blueprint:
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    return blueprint


def log_world_frame_and_pcd(pcd: o3d.geometry.PointCloud, pcd_colormap: str):
    # log world frame
    rr.log(
        f"world_frame",
        rr.Transform3D(axis_length=1.0),
        static=True,
    )

    # load and log pcd
    pcd_points = np.asarray(pcd.points)
    pcd_points_z = pcd_points[:, -1]
    pcd_points_min_z = pcd_points_z.min()
    pcd_points_max_z = pcd_points_z.max()
    pcd_z_norm = matplotlib.colors.Normalize(
        vmin=pcd_points_min_z, vmax=pcd_points_max_z
    )

    pcd_cmap = matplotlib.colormaps[pcd_colormap]
    rr.log(
        f"pcd",
        rr.Points3D(pcd_points, colors=pcd_cmap(pcd_z_norm(pcd_points_z))),
        static=True,
    )


def log_waypoint_data(
    wp_p: torch.Tensor,
    wp_q: torch.Tensor,
    wp_w: torch.Tensor,
    wp_h: torch.Tensor,
):
    num_waypoints = wp_p.shape[0]
    for i in range(num_waypoints):
        rr.log(
            f"waypoint/{i}",
            rr.Boxes3D(
                half_sizes=[0.05, wp_w[i] / 2, wp_h[i] / 2],
                labels=f"{i}",
                colors=[0, 255, 0],
            ),
            static=True,
        )
        rr.log(
            f"waypoint/{i}",
            rr.Transform3D(
                translation=wp_p[i],
                rotation=rr.Quaternion(xyzw=wp_q[i]),
                axis_length=1.0,
            ),
            static=True,
        )
        if i < num_waypoints - 1:
            rr.log(
                f"waypoint/line_segment/{i}_{i + 1}",
                rr.LineStrips3D(wp_p[i : i + 2].numpy(), colors=[0, 255, 0]),
                static=True,
            )


def log_episode_data(
    ep_dict: Dict[str, Any],
    cam_params: Dict[str, Any],
    vel_colormap: str,
    vel_max_cmap: float,
    traj_line_weight: float,
    drone_urdf: str,
    num_episodes: int,
    log_cam: bool,
    ep_prefix: str,
    only_traj: bool = False,
):
    num_substeps = ep_dict["ep_0"]["position_w"][0].shape[0]
    step_dt = ep_dict["ep_0"]["t"][1]
    sim_dt = step_dt / num_substeps
    vel_cmap = matplotlib.colormaps[vel_colormap]

    # load urdf
    urdf_logger = None
    if drone_urdf is not None:
        urdf_logger = URDFLogger(drone_urdf, None)

    for i in range(num_episodes):
        num_steps = len(ep_dict[f"ep_{i}"]["t"])

        # log trajectory
        pos_tensor = torch.stack(ep_dict[f"ep_{i}"]["position_w"]).flatten(0, 1)
        line_start = pos_tensor[:-1]  # (N-1, 3)
        line_end = pos_tensor[1:]  # (N-1, 3)
        line_data = torch.stack((line_start, line_end), dim=1).numpy()  # (N-1, 2, 3)
        ep_vel_norm = (
            torch.stack(ep_dict[f"ep_{i}"]["lin_vel_w"]).flatten(0, 1).norm(dim=1)
        )
        vel_line_avg = ((ep_vel_norm[:-1] + ep_vel_norm[1:]) / 2).numpy()
        rr.log(
            f"episode_{ep_prefix}{i}/trajectory",
            rr.LineStrips3D(
                line_data,
                colors=vel_cmap(vel_line_avg / vel_max_cmap),
                radii=rr.Radius.ui_points(traj_line_weight),
            ),
            static=True,
        )

        if not only_traj:
            # log urdf
            if urdf_logger is not None:
                urdf_logger.entity_path_prefix = f"episode_{ep_prefix}{i}/urdf"
                urdf_logger.log()

            # log time series line indicator
            for field in ["ang_vel_des_b_frd", "ang_vel_b_frd", "lin_vel_b_frd"]:
                for dim in ["x", "y", "z"]:
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/{field}/{dim}",
                        rr.SeriesLine(name=f"{field}_{dim}"),
                        static=True,
                    )
            rr.log(
                f"episode_{ep_prefix}{i}/scalar/speed",
                rr.SeriesLine(name="speed"),
                static=True,
            )
            for rotor_id in range(4):
                rr.log(
                    f"episode_{ep_prefix}{i}/scalar/rotor_cmd/{rotor_id}",
                    rr.SeriesLine(name=f"rotor_cmd_{rotor_id}"),
                    static=True,
                )
            rr.log(
                f"episode_{ep_prefix}{i}/scalar/min_d_to_obst",
                rr.SeriesLine(name="min_d_to_obst"),
                static=True,
            )

            for j in range(num_steps):
                step_t = ep_dict[f"ep_{i}"]["t"][j]
                step_min_d_obst = ep_dict[f"ep_{i}"]["min_dist_to_obstacle"][j]
                step_wp_pos = ep_dict[f"ep_{i}"]["next_waypoint_p"][j]
                # TODO: log action?

                step_depth = None
                step_color = None
                step_cam_pose = None
                if log_cam:
                    step_depth = ep_dict[f"ep_{i}"]["main_depth"][j]
                    step_color = ep_dict[f"ep_{i}"]["main_color"][j]
                    step_cam_pose = ep_dict[f"ep_{i}"]["main_cam_pose"][j]

                for k in range(num_substeps):
                    substep_t = step_t + sim_dt * k
                    substep_ang_vel_d_b_frd = ep_dict[f"ep_{i}"]["ang_vel_des_b_frd"][
                        j
                    ][k]
                    substep_rotor_cmd = ep_dict[f"ep_{i}"]["rotor_cmd"][j][k]
                    substep_pos = ep_dict[f"ep_{i}"]["position_w"][j][k]
                    substep_quat = ep_dict[f"ep_{i}"]["quaternion_w"][j][k]
                    substep_lin_vel_w = ep_dict[f"ep_{i}"]["lin_vel_w"][j][k]
                    substep_lin_vel_b_frd = ep_dict[f"ep_{i}"]["lin_vel_b_frd"][j][k]
                    substep_ang_vel_b_frd = ep_dict[f"ep_{i}"]["ang_vel_b_frd"][j][k]

                    substep_lin_vel_norm = substep_lin_vel_w.norm()
                    vel_mapped_color = vel_cmap(substep_lin_vel_norm / vel_max_cmap)

                    # log time
                    rr.set_time_seconds("sim_time", substep_t)

                    # log position
                    rr.log(
                        f"episode_{ep_prefix}{i}/position",
                        rr.Points3D(
                            substep_pos,
                            colors=vel_mapped_color,
                        ),
                    )

                    # log timeseries
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_des_b_frd/x",
                        rr.Scalar(substep_ang_vel_d_b_frd[0]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_des_b_frd/y",
                        rr.Scalar(substep_ang_vel_d_b_frd[1]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_des_b_frd/z",
                        rr.Scalar(substep_ang_vel_d_b_frd[2]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_b_frd/x",
                        rr.Scalar(substep_ang_vel_b_frd[0]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_b_frd/y",
                        rr.Scalar(substep_ang_vel_b_frd[1]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/ang_vel_b_frd/z",
                        rr.Scalar(substep_ang_vel_b_frd[2]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/lin_vel_b_frd/x",
                        rr.Scalar(substep_lin_vel_b_frd[0]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/lin_vel_b_frd/y",
                        rr.Scalar(substep_lin_vel_b_frd[1]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/lin_vel_b_frd/z",
                        rr.Scalar(substep_lin_vel_b_frd[2]),
                        rr.SeriesLine(),
                    )
                    rr.log(
                        f"episode_{ep_prefix}{i}/scalar/speed",
                        rr.Scalar(substep_lin_vel_norm),
                        rr.SeriesLine(),
                    )
                    for rotor_id in range(4):
                        rr.log(
                            f"episode_{ep_prefix}{i}/scalar/rotor_cmd/{rotor_id}",
                            rr.Scalar(substep_rotor_cmd[rotor_id]),
                            rr.SeriesLine(),
                        )

                    # log low frequency data
                    if k == 0:
                        # log min dist to obstacle
                        rr.log(
                            f"episode_{ep_prefix}{i}/scalar/min_d_to_obst",
                            rr.Scalar(step_min_d_obst),
                            rr.SeriesLine(),
                        )

                        # log lin vel vector
                        rr.log(
                            f"episode_{ep_prefix}{i}/velocity",
                            rr.Arrows3D(
                                origins=substep_pos,
                                vectors=substep_lin_vel_w / substep_lin_vel_norm,
                                colors=vel_mapped_color,
                            ),
                        )

                        # log vector to target waypoint
                        rr.log(
                            f"episode_{ep_prefix}{i}/vec_to_wp",
                            rr.Arrows3D(
                                origins=substep_pos,
                                vectors=step_wp_pos - substep_pos,
                                colors=[0, 255, 0],
                            ),
                        )

                        # log camera
                        if log_cam:
                            tf_body_to_cam = torch.eye(4)
                            tf_body_to_cam[:3, :3] = step_cam_pose[3:].reshape(3, 3)
                            tf_body_to_cam[:3, 3] = step_cam_pose[:3]
                            tf_world_to_body = torch.eye(4)
                            tf_world_to_body[:3, :3] = torch.tensor(
                                o3d.geometry.get_rotation_matrix_from_quaternion(
                                    substep_quat.roll(1).numpy()
                                )
                            )
                            tf_world_to_body[:3, 3] = substep_pos
                            tf_world_to_cam = tf_world_to_body @ tf_body_to_cam
                            rr.log(
                                f"episode_{ep_prefix}{i}/camera",
                                rr.Pinhole(
                                    focal_length=float(cam_params["f"]),
                                    width=int(cam_params["w"]),
                                    height=int(cam_params["h"]),
                                    camera_xyz=rr.ViewCoordinates.FLU,
                                    image_plane_distance=1.0,
                                ),
                            )
                            rr.log(
                                f"episode_{ep_prefix}{i}/camera/color",
                                rr.Image(step_color),
                            )
                            rr.log(
                                f"episode_{ep_prefix}{i}/camera/depth",
                                rr.DepthImage(
                                    step_depth,
                                    meter=(1 / cam_params["depth_scale"]),
                                    colormap=rr.components.Colormap(1),  # gray scale
                                ),
                            )
                            rr.log(
                                f"episode_{ep_prefix}{i}/camera",
                                rr.Transform3D(
                                    translation=tf_world_to_cam[:3, 3],
                                    mat3x3=tf_world_to_cam[:3, :3],
                                    axis_length=0.0,
                                ),
                            )

                        # log urdf transform
                        rr.log(
                            f"episode_{ep_prefix}{i}/urdf",
                            rr.Transform3D(
                                translation=substep_pos,
                                rotation=rr.Quaternion(xyzw=substep_quat),
                                axis_length=1.0,
                            ),
                        )


@rr.shutdown_at_exit
def proc_env(
    only_calc_metrics: bool,
    combine: bool,
    only_traj_combine: bool,
    combine_rec_id: str,
    cam_params: Dict[str, Any],
    pcd_colormap: str,
    vel_colormap: str,
    vel_max_cmap: float,
    traj_line_weight: float,
    drone_urdf: str,
    exp_dir: str,
    env_id: int,
    num_episodes: int,
    wp_data_p: torch.Tensor,  # (num_wps, 3)
    wp_data_q: torch.Tensor,  # (num_wps, 4)
    wp_data_w: torch.Tensor,  # (num_wps, )
    wp_data_h: torch.Tensor,  # (num_wps, )
) -> Dict[str, Any]:
    # start stopwatch
    t_start = time.time()

    # load episode data from file
    ep_dict: Dict = torch.load(os.path.join(exp_dir, f"log_{env_id}.pt"))

    rrd_file = "no rrd file as only calculating the metrics"
    if not only_calc_metrics:
        # init rerun
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        rrd_file = os.path.join(exp_dir, f"env_{env_id}.rrd")
        rr.init(application_id=exp_name, recording_id=f"env_{env_id}")
        rr.save(path=rrd_file, default_blueprint=rrb_single_env())

        # load pcd
        pcd = o3d.io.read_point_cloud(os.path.join(exp_dir, f"pcd_{env_id}.ply"))

        # log world frame and pcd
        log_world_frame_and_pcd(pcd, pcd_colormap)

        # log waypoint data
        log_waypoint_data(wp_data_p, wp_data_q, wp_data_w, wp_data_h)

        # log episode data
        log_episode_data(
            ep_dict,
            cam_params,
            vel_colormap,
            vel_max_cmap,
            traj_line_weight,
            drone_urdf,
            num_episodes,
            True,
            "",
        )

        # log extra data if combine envs
        if combine:
            rr.init(application_id=exp_name, recording_id=combine_rec_id)
            rr.connect()
            log_world_frame_and_pcd(pcd, pcd_colormap)
            log_waypoint_data(wp_data_p, wp_data_q, wp_data_w, wp_data_h)
            log_episode_data(
                ep_dict,
                cam_params,
                vel_colormap,
                vel_max_cmap,
                traj_line_weight,
                drone_urdf,
                num_episodes,
                False,
                str(env_id),
                only_traj_combine,
            )

    # calculate metrics for this env
    metrics: Dict[str, Any] = {
        "num_finishes": 0,
        "num_crashes": 0,
        "num_timeouts": 0,
        "min_safety_margins": [],  # each item correspond to one episode
        "avg_lin_speeds": [],
        "max_lin_speeds": [],
        "avg_ang_speeds": [],
        "max_ang_speeds": [],
        "avg_avg_rotor_cmds": [],
        "max_avg_rotor_cmds": [],
    }
    for i in range(num_episodes):
        # termination flag
        is_finished = ep_dict[f"ep_{i}"]["is_finished"]
        is_crashed = ep_dict[f"ep_{i}"]["is_crashed"]
        is_timeout = ep_dict[f"ep_{i}"]["is_timeout"]
        assert len(is_finished) == len(is_crashed) == len(is_timeout)

        ep_finished = torch.any(torch.stack(is_finished))
        ep_crashed = torch.any(torch.stack(is_crashed))
        ep_timeout = torch.any(torch.stack(is_timeout))
        assert ep_finished.int() + ep_crashed.int() + ep_timeout.int() == 1

        if ep_finished:
            metrics["num_finishes"] += 1
        elif ep_crashed:
            metrics["num_crashes"] += 1
        elif ep_timeout:
            metrics["num_timeouts"] += 1

        # safety margin
        min_dist_to_obstacle = ep_dict[f"ep_{i}"]["min_dist_to_obstacle"]
        metrics["min_safety_margins"].append(
            float(torch.min(torch.stack(min_dist_to_obstacle)))
        )

        # lin speed
        lin_vel = ep_dict[f"ep_{i}"]["lin_vel_w"]
        if ep_crashed:
            lin_vel.pop()
        lin_speed = torch.linalg.norm(torch.cat(lin_vel), dim=1)
        metrics["avg_lin_speeds"].append(float(torch.mean(lin_speed)))
        metrics["max_lin_speeds"].append(float(torch.max(lin_speed)))

        # ang speed
        ang_vel = ep_dict[f"ep_{i}"]["ang_vel_b_frd"]
        if ep_crashed:
            ang_vel.pop()
        ang_speed = torch.linalg.norm(torch.cat(ang_vel), dim=1)
        metrics["avg_ang_speeds"].append(float(torch.mean(ang_speed)))
        metrics["max_ang_speeds"].append(float(torch.max(ang_speed)))

        # rotor cmd
        rotor_cmd = ep_dict[f"ep_{i}"]["rotor_cmd"]
        if ep_crashed:
            rotor_cmd.pop()
        avg_rotor_cmd = torch.cat(rotor_cmd).mean(dim=1)
        metrics["avg_avg_rotor_cmds"].append(float(torch.mean(avg_rotor_cmd)))
        metrics["max_avg_rotor_cmds"].append(float(torch.max(avg_rotor_cmd)))

    assert (
        metrics["num_finishes"] + metrics["num_crashes"] + metrics["num_timeouts"]
        == num_episodes
    )

    # end stopwatch
    t_end = time.time()
    print(
        f"[process env] env {env_id}, process time {t_end - t_start}, created {rrd_file}"
    )

    # return the metrics for this env
    return metrics


def main():
    # info
    print("+++ Rerunning experiment")

    # Suppress torch.load warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", type=str, required=True)
    arg_parser.add_argument("--num_processes", type=int, default=16)
    arg_parser.add_argument("--pcd_colormap", type=str, default="turbo")
    arg_parser.add_argument("--vel_colormap", type=str, default="plasma")
    arg_parser.add_argument("--vel_max_cmap", type=float, default=25.0)
    arg_parser.add_argument("--traj_line_weight", type=float, default=0.5)
    arg_parser.add_argument("--drone_urdf", type=str)
    arg_parser.add_argument("--combine_envs", action="store_true")
    arg_parser.add_argument("--only_calc_metrics", action="store_true")
    arg_parser.add_argument("--only_traj_combine", action="store_true")

    args = arg_parser.parse_args()
    exp_dir: str = args.exp_dir
    num_processes: int = args.num_processes
    pcd_colormap: str = args.pcd_colormap
    vel_colormap: str = args.vel_colormap
    vel_max_cmap: float = args.vel_max_cmap
    traj_line_weight: float = args.traj_line_weight
    drone_urdf: str = args.drone_urdf
    combine_envs: bool = args.combine_envs
    only_calc_metrics: bool = args.only_calc_metrics
    only_traj_combine: bool = args.only_traj_combine

    # get info from cfg
    num_envs = 0
    cam_params: Dict[str, Any] = {}
    num_episodes = 0
    wp_data_p_list: List[torch.Tensor] = []
    wp_data_q_list: List[torch.Tensor] = []
    wp_data_w_list: List[torch.Tensor] = []
    wp_data_h_list: List[torch.Tensor] = []

    log_dirs: List[str] = [
        os.path.join(exp_dir, d)
        for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    ]
    log_dirs.sort()
    for log_dir in log_dirs:
        cfg: Dict[str, Any] = torch.load(os.path.join(log_dir, "cfg.pt"))
        num_envs += cfg["env"]["numEnvs"]
        wp_data_p_list.append(cfg["waypoint_data_p"])
        wp_data_q_list.append(cfg["waypoint_data_q"])
        wp_data_w_list.append(cfg["waypoint_data_w"])
        wp_data_h_list.append(cfg["waypoint_data_h"])
        if len(cam_params) == 0:
            w = cfg["env"]["cameraWidth"]
            h = cfg["env"]["cameraHeight"]
            hfov = cfg["env"]["cameraHfov"]
            max_depth = cfg["env"]["cameraDepthMax"]
            f = w / (2 * np.tan(np.deg2rad(hfov) / 2))
            cam_params["w"] = w
            cam_params["h"] = h
            cam_params["f"] = f
            cam_params["depth_scale"] = max_depth
            num_episodes = cfg["env"]["logging"]["maxNumEpisodes"]
            print(cam_params)
            print(f"number of episodes: {num_episodes}")

    # we assume the number of waypoints is constant throughout one experiment
    wp_data_p = torch.cat(wp_data_p_list)  # (num_envs, num_wps, 3)
    wp_data_q = torch.cat(wp_data_q_list)  # (num_envs, num_wps, 4)
    wp_data_w = torch.cat(wp_data_w_list)  # (num_envs, num_wps)
    wp_data_h = torch.cat(wp_data_h_list)  # (num_envs, num_wps)
    assert wp_data_p.shape[0] == num_envs
    assert wp_data_q.shape[0] == num_envs
    assert wp_data_w.shape[0] == num_envs
    assert wp_data_h.shape[0] == num_envs

    # extra initialization if we need to combine envs
    combine_rec_id: str = "env_combined_" + "{date:%y-%m-%d-%H-%M-%S}".format(
        date=datetime.now()
    )
    if combine_envs:
        rr.init(
            application_id=os.path.basename(os.path.normpath(exp_dir)),
            recording_id=combine_rec_id,
            spawn=True,
        )

    # process data for envs in parallel
    print("processing data in parallel")
    with mp.Pool(min(num_processes, num_envs)) as pool:
        env_metrics: List[Dict[str, Any]] = pool.starmap(
            proc_env,
            [
                (
                    only_calc_metrics,
                    combine_envs,
                    only_traj_combine,
                    combine_rec_id,
                    cam_params,
                    pcd_colormap,
                    vel_colormap,
                    vel_max_cmap,
                    traj_line_weight,
                    drone_urdf,
                    exp_dir,
                    env_id,
                    num_episodes,
                    wp_data_p[env_id].clone(),
                    wp_data_q[env_id].clone(),
                    wp_data_w[env_id].clone(),
                    wp_data_h[env_id].clone(),
                )
                for env_id in range(num_envs)
            ],
        )

    if combine_envs:
        # TODO: automatic saving
        rr.send_blueprint(rrb_combined_env())
        print("[main] please manually save rr data for combined env")

    # now we have a list of dictionaries containing episode data
    # aggregate them and save them
    aggregated_metrics: Dict[str, Any] = {
        "num_finishes": 0,
        "num_crashes": 0,
        "num_timeouts": 0,
        "min_safety_margins": [],  # each item correspond to one episode
        "avg_lin_speeds": [],
        "max_lin_speeds": [],
        "avg_ang_speeds": [],
        "max_ang_speeds": [],
        "avg_avg_rotor_cmds": [],
        "max_avg_rotor_cmds": [],
    }

    for i in range(num_envs):
        metrics = env_metrics[i]
        aggregated_metrics["num_finishes"] += metrics["num_finishes"]
        aggregated_metrics["num_crashes"] += metrics["num_crashes"]
        aggregated_metrics["num_timeouts"] += metrics["num_timeouts"]
        aggregated_metrics["min_safety_margins"].extend(metrics["min_safety_margins"])
        aggregated_metrics["avg_lin_speeds"].extend(metrics["avg_lin_speeds"])
        aggregated_metrics["max_lin_speeds"].extend(metrics["max_lin_speeds"])
        aggregated_metrics["avg_ang_speeds"].extend(metrics["avg_ang_speeds"])
        aggregated_metrics["max_ang_speeds"].extend(metrics["max_ang_speeds"])
        aggregated_metrics["avg_avg_rotor_cmds"].extend(metrics["avg_avg_rotor_cmds"])
        aggregated_metrics["max_avg_rotor_cmds"].extend(metrics["max_avg_rotor_cmds"])

    total_episodes = num_envs * num_episodes
    assert (
        aggregated_metrics["num_finishes"]
        + aggregated_metrics["num_crashes"]
        + aggregated_metrics["num_timeouts"]
        == total_episodes
    )
    assert len(aggregated_metrics["min_safety_margins"]) == total_episodes
    assert len(aggregated_metrics["avg_lin_speeds"]) == total_episodes
    assert len(aggregated_metrics["max_lin_speeds"]) == total_episodes
    assert len(aggregated_metrics["avg_ang_speeds"]) == total_episodes
    assert len(aggregated_metrics["max_ang_speeds"]) == total_episodes
    assert len(aggregated_metrics["avg_avg_rotor_cmds"]) == total_episodes
    assert len(aggregated_metrics["max_avg_rotor_cmds"]) == total_episodes
    torch.save(aggregated_metrics, os.path.join(exp_dir, "metrics.pt"))


if __name__ == "__main__":
    main()
