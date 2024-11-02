import argparse
import gc
import multiprocessing as mp
import os
import time
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import open3d as o3d
import torch


def empty_log_items_dict() -> Dict[str, List[Any]]:
    return {
        "t": [],
        "main_depth": [],
        "main_color": [],
        "min_dist_to_obstacle": [],
        "main_cam_pose": [],
        "action": [],
        "next_waypoint_p": [],
        "ang_vel_des_b_frd": [],
        "rotor_cmd": [],
        "position_w": [],
        "quaternion_w": [],
        "lin_vel_w": [],
        "lin_vel_b_frd": [],
        "ang_vel_b_frd": [],
        "is_finished": [],
        "is_crashed": [],
        "is_timeout": [],
    }


def pcd_from_np_array(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def convert_o3d(mat: np.ndarray) -> np.ndarray:
    ret = mat.copy()
    ret[:3, 0] = -mat[:3, 1]
    ret[:3, 1] = -mat[:3, 2]
    ret[:3, 2] = mat[:3, 0]
    return ret


def pinhole_depth_to_pcd(
    depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic, tf: np.ndarray
) -> o3d.geometry.PointCloud:
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d.geometry.Image(depth),
        intrinsic=intrinsic,
    )  # noqa
    return pcd.transform(tf)


def process_log(
    global_env_id: int,
    num_substeps: int,
    sim_dt: float,
    pcd_proc_params: Dict[str, Any],
    ep_dict: Dict[str, Any],  # dict containing keys "ep_0", "ep_1", ...
    pcd_points: np.ndarray,
    env_step: List[int],
    episode_id: torch.Tensor,  # (num_envs, )
    episode_progress: torch.Tensor,  # (num_steps, )
    main_depth: torch.Tensor,  # (num_steps, h, w)
    main_color: torch.Tensor,  # (num_steps, h, w, 4)
    extra_depth: torch.Tensor,  # (num_steps, 6, cam_h, cam_w)
    min_dist_to_obstacle: torch.Tensor,  # (num_steps, )
    main_cam_pose: torch.Tensor,  # (num_steps, 12)
    action: torch.Tensor,  # (num_steps, 4)
    next_waypoint_p: torch.Tensor,  # (num_steps, 3)
    ang_vel_des_b_frd: torch.Tensor,  # (num_steps, ctrl_freq_inv, 3)
    rotor_cmd: torch.Tensor,  # (num_steps, ctrl_freq_inv, 4)
    position_w: torch.Tensor,  # (num_steps, ctrl_freq_inv, 3)
    quaternion_w: torch.Tensor,  # (num_steps, ctrl_freq_inv, 4)
    lin_vel_w: torch.Tensor,  # (num_steps, ctrl_freq_inv, 3)
    lin_vel_b_frd: torch.Tensor,  # (num_steps, ctrl_freq_inv, 3)
    ang_vel_b_frd: torch.Tensor,  # (num_steps, ctrl_freq_inv, 3)
    is_finished: torch.Tensor,  # (num_steps, )
    is_crashed: torch.Tensor,  # (num_steps, )
    is_timeout: torch.Tensor,  # (num_steps, )
) -> Tuple[Dict, np.ndarray]:
    # get starting time
    t_start = time.time()

    # prepare for pcd integration
    pcd = pcd_from_np_array(pcd_points)
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=pcd_proc_params["w"],
        height=pcd_proc_params["h"],
        fx=pcd_proc_params["fx"],
        fy=pcd_proc_params["fy"],
        cx=pcd_proc_params["cx"],
        cy=pcd_proc_params["cy"],
    )
    q_w_first = quaternion_w[:, 0].roll(1, 1)
    pcd_update_itv = pcd_proc_params["pcd_update_itv"]

    # calculate dt between two steps
    step_dt = sim_dt * num_substeps

    # iterate through steps and put each step into the right episode
    num_steps = len(env_step)
    for i in range(num_steps):
        # identify episode
        step_ep_id = int(episode_id[i])
        ep_name = f"ep_{step_ep_id}"
        if not ep_name in ep_dict:
            ep_dict[ep_name] = empty_log_items_dict()

        # get timestamp from episode progress
        step_ep_prog = float(episode_progress[i])
        step_t = step_ep_prog * step_dt

        # feed data into the lists
        ep_dict[ep_name]["t"].append(step_t)
        ep_dict[ep_name]["main_depth"].append(main_depth[i])
        ep_dict[ep_name]["main_color"].append(main_color[i])
        ep_dict[ep_name]["min_dist_to_obstacle"].append(min_dist_to_obstacle[i])
        ep_dict[ep_name]["main_cam_pose"].append(main_cam_pose[i])
        ep_dict[ep_name]["action"].append(action[i])
        ep_dict[ep_name]["next_waypoint_p"].append(next_waypoint_p[i])
        ep_dict[ep_name]["ang_vel_des_b_frd"].append(ang_vel_des_b_frd[i])
        ep_dict[ep_name]["rotor_cmd"].append(rotor_cmd[i])
        ep_dict[ep_name]["position_w"].append(position_w[i])
        ep_dict[ep_name]["quaternion_w"].append(quaternion_w[i])
        ep_dict[ep_name]["lin_vel_w"].append(lin_vel_w[i])
        ep_dict[ep_name]["lin_vel_b_frd"].append(lin_vel_b_frd[i])
        ep_dict[ep_name]["ang_vel_b_frd"].append(ang_vel_b_frd[i])
        ep_dict[ep_name]["is_finished"].append(is_finished[i])
        ep_dict[ep_name]["is_crashed"].append(is_crashed[i])
        ep_dict[ep_name]["is_timeout"].append(is_timeout[i])

        # process extra depth into pcd
        if i % pcd_update_itv == 0:
            # extra depth images front the batch
            depth_front = extra_depth[i, 0].numpy()
            depth_back = extra_depth[i, 1].numpy()
            depth_left = extra_depth[i, 2].numpy()
            depth_right = extra_depth[i, 3].numpy()
            depth_up = extra_depth[i, 4].numpy()
            depth_down = extra_depth[i, 5].numpy()

            # transform matrices of cameras
            q_front = q_w_first[i].numpy()

            mat_front: np.ndarray = np.eye(4)
            mat_front[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
                q_front
            )
            mat_front[:3, 3] = position_w[i, 0].numpy()

            mat_back = mat_front.copy()
            mat_back[:3, :2] *= -1

            mat_left = mat_front.copy()
            mat_left[:3, 0] = mat_front[:3, 1]
            mat_left[:3, 1] = -mat_front[:3, 0]

            mat_right = mat_left.copy()
            mat_right[:3, :2] *= -1

            mat_up = mat_front.copy()
            mat_up[:3, 0] = mat_front[:3, 2]
            mat_up[:3, 2] = -mat_front[:3, 0]

            mat_down = mat_up.copy()
            mat_down[:3, [0, 2]] *= -1

            # create pcds
            pcd_front = pinhole_depth_to_pcd(
                depth_front, cam_intrinsic, convert_o3d(mat_front)
            )
            pcd_back = pinhole_depth_to_pcd(
                depth_back, cam_intrinsic, convert_o3d(mat_back)
            )
            pcd_left = pinhole_depth_to_pcd(
                depth_left, cam_intrinsic, convert_o3d(mat_left)
            )
            pcd_right = pinhole_depth_to_pcd(
                depth_right, cam_intrinsic, convert_o3d(mat_right)
            )
            pcd_up = pinhole_depth_to_pcd(depth_up, cam_intrinsic, convert_o3d(mat_up))
            pcd_down = pinhole_depth_to_pcd(
                depth_down, cam_intrinsic, convert_o3d(mat_down)
            )
            pcd += pcd_front + pcd_back + pcd_left + pcd_right + pcd_up + pcd_down

    # down-sample the pcd before returning
    pcd = pcd.voxel_down_sample(pcd_proc_params["voxel_size"])

    # get ending time and print info
    t_end = time.time()
    print(f"[process log] env {global_env_id}, process time {t_end - t_start} s, {pcd}")

    return ep_dict, np.asarray(pcd.points)


def save_data(
    global_env_id: int,
    exp_dir: str,
    ep_dict: Dict,
    pcd_points: np.ndarray,
):
    t_start = time.time()

    torch.save(ep_dict, os.path.join(exp_dir, f"log_{global_env_id}.pt"))
    o3d.io.write_point_cloud(
        os.path.join(exp_dir, f"pcd_{global_env_id}.ply"), pcd_from_np_array(pcd_points)
    )

    t_end = time.time()
    print(f"[save data] env {global_env_id}, process time {t_end - t_start} s")


def main():
    # info
    print("+++ Processing log files")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", type=str, required=True)
    arg_parser.add_argument("--num_processes", type=int, default=16)
    arg_parser.add_argument("--voxel_size", type=float, default=0.05)
    arg_parser.add_argument("--pcd_update_itv", type=int, default=25)

    args = arg_parser.parse_args()
    exp_dir: str = args.exp_dir
    num_processes: int = args.num_processes
    voxel_size: float = args.voxel_size
    pcd_update_itv: int = args.pcd_update_itv

    # get all log dirs
    log_dirs: List[str] = [
        os.path.join(exp_dir, d)
        for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    ]
    log_dirs.sort()
    print("log dirs include:")
    print(log_dirs)
    print(
        "make sure there are no other dirs in the experiment dir, "
        "otherwise this script will not run"
    )

    # get total number of envs (separated in different log dirs)
    num_total_envs = 0
    for log_dir in log_dirs:
        cfg: Dict[str, Any] = torch.load(os.path.join(log_dir, "cfg.pt"))
        num_total_envs += cfg["env"]["numEnvs"]

    # dictionary of data for all envs
    # env_episode_data["env_0"] is for env[0], and is a dict containing keys "ep_0", "ep_1", ...
    # env_episode_data["env_0"]["ep_0"] is a dictionary of specific data items like position
    # each data item is eventually a list representing data on the timeline
    # env_pcd_points[i] is a numpy array storing points for env[i]
    env_episode_data: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
        f"env_{i}": {} for i in range(num_total_envs)
    }
    env_pcd_points: Dict[str, np.ndarray] = {
        f"env_{i}": np.empty((0, 3)) for i in range(num_total_envs)
    }

    # iterate through log dirs to update env data dict and env pcd points
    env_id_offset: int = 0
    for log_dir in log_dirs:
        cfg: Dict[str, Any] = torch.load(os.path.join(log_dir, "cfg.pt"))
        num_envs_log: int = cfg["env"]["numEnvs"]
        num_log_files: int = cfg["env"]["logging"]["numLogFiles"]
        ctrl_freq_inv: int = cfg["env"]["controlFrequencyInv"]
        sim_dt: float = cfg["sim"]["dt"]
        cam_w: int = cfg["env"]["logging"]["extraCameraWidth"]
        cam_h: int = cfg["env"]["logging"]["extraCameraHeight"]
        cam_hfov: float = cfg["env"]["logging"]["extraCameraHfov"]

        # pcd process params
        cam_fx = cam_w / (2 * np.tan(np.deg2rad(cam_hfov) / 2))
        cam_fy = cam_fx * cam_h / cam_w
        cam_cx = cam_w / 2
        cam_cy = cam_h / 2
        pcd_proc_params: Dict[str, Any] = {
            "w": cam_w,
            "h": cam_h,
            "fx": cam_fx,
            "fy": cam_fy,
            "cx": cam_cx,
            "cy": cam_cy,
            "voxel_size": voxel_size,
            "pcd_update_itv": pcd_update_itv,
        }

        # process all log files
        for i in range(num_log_files):
            log_file = os.path.join(log_dir, str(i) + ".pt")
            print(f"loading {log_file}")
            file_dict: Dict[str, Any] = torch.load(log_file)
            print("done loading")

            # extract info, stack tensors so we can select slices for each env
            print("extracting info from dict")
            env_step: List[int] = file_dict["env_step"]
            episode_id = torch.stack(file_dict["episode_id"])
            episode_progress = torch.stack(file_dict["episode_progress"])
            main_depth = torch.stack(file_dict["main_depth"])
            main_color = torch.stack(file_dict["main_color"])
            extra_depth = torch.stack(file_dict["extra_depth"])
            min_dist_to_obstacle = torch.stack(file_dict["min_dist_to_obstacle"])
            main_cam_pose = torch.stack(file_dict["main_cam_pose"])
            action = torch.stack(file_dict["action"])
            next_waypoint_p = torch.stack(file_dict["next_waypoint_p"])
            ang_vel_des_b_frd = torch.stack(file_dict["ang_vel_des_b_frd"])
            rotor_cmd = torch.stack(file_dict["rotor_cmd"])
            position_w = torch.stack(file_dict["position_w"])
            quaternion_w = torch.stack(file_dict["quaternion_w"])
            lin_vel_w = torch.stack(file_dict["lin_vel_w"])
            lin_vel_b_frd = torch.stack(file_dict["lin_vel_b_frd"])
            ang_vel_b_frd = torch.stack(file_dict["ang_vel_b_frd"])
            is_finished = torch.stack(file_dict["is_finished"])
            is_crashed = torch.stack(file_dict["is_crashed"])
            is_timeout = torch.stack(file_dict["is_timeout"])
            file_dict.clear()  # free up some mem
            gc.collect()
            print("done extracting info")

            # process log file
            print(f"processing log file {log_file}")
            with mp.Pool(min(num_processes, num_envs_log)) as pool:
                ret: List[Tuple[Dict, np.ndarray]] = pool.starmap(
                    process_log,
                    [
                        (
                            env_id + env_id_offset,  # global env id
                            ctrl_freq_inv,
                            sim_dt,
                            pcd_proc_params,
                            env_episode_data[f"env_{env_id + env_id_offset}"],
                            env_pcd_points[f"env_{env_id + env_id_offset}"],
                            env_step,
                            episode_id[:, env_id].clone(),
                            episode_progress[:, env_id].clone(),
                            main_depth[:, env_id].clone(),
                            main_color[:, env_id].clone(),
                            extra_depth[:, env_id].clone(),
                            min_dist_to_obstacle[:, env_id].clone(),
                            main_cam_pose[:, env_id].clone(),
                            action[:, env_id].clone(),
                            next_waypoint_p[:, env_id].clone(),
                            ang_vel_des_b_frd[:, :, env_id].clone(),
                            rotor_cmd[:, :, env_id].clone(),
                            position_w[:, :, env_id].clone(),
                            quaternion_w[:, :, env_id].clone(),
                            lin_vel_w[:, :, env_id].clone(),
                            lin_vel_b_frd[:, :, env_id].clone(),
                            ang_vel_b_frd[:, :, env_id].clone(),
                            is_finished[:, env_id].clone(),
                            is_crashed[:, env_id].clone(),
                            is_timeout[:, env_id].clone(),
                        )
                        for env_id in range(num_envs_log)
                    ],
                )

            # update env data dict and env pcd using ret
            for env_id in range(num_envs_log):
                global_env_id = env_id + env_id_offset
                (
                    env_episode_data[f"env_{global_env_id}"],
                    env_pcd_points[f"env_{global_env_id}"],
                ) = ret[env_id]

        # save env data
        print(f"saving data extracted from {log_dir}")
        with mp.Pool(min(num_processes, num_envs_log)) as pool:
            pool.starmap(
                save_data,
                [
                    (
                        env_id + env_id_offset,
                        exp_dir,
                        env_episode_data[f"env_{env_id + env_id_offset}"],
                        env_pcd_points[f"env_{env_id + env_id_offset}"],
                    )
                    for env_id in range(num_envs_log)
                ],
            )

        # clear already saved data
        for env_id in range(num_envs_log):
            global_env_id = env_id + env_id_offset
            env_episode_data[f"env_{global_env_id}"].clear()
            env_pcd_points[f"env_{global_env_id}"] = np.empty((0, 3))
        gc.collect()

        # update global env id offset
        env_id_offset += num_envs_log


if __name__ == "__main__":
    main()
