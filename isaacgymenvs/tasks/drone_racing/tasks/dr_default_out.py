import abc

import torch

from .dr_base import DRBase


class DRDefaultOut(DRBase):

    @abc.abstractmethod
    def _create_envs(self):
        pass

    @abc.abstractmethod
    def _randomize_racing_tracks(self):
        pass

    def _update_rew_buf(self):
        self.lin_vel_reward = (
            self.k_vel_lateral_rew * self.drone_lin_vel_b_frd[:, 1] ** 2
            + self.k_vel_backward_rew * self.drone_lin_vel_b_frd[:, 0].clamp(max=0) ** 2
        )
        self.rew_buf[:] = self.default_reward + self.lin_vel_reward

    def _update_extra_rew_terms(self):
        self.extras["reward_progress"] = self.mdp_reward.reward_progress
        self.extras["reward_perception"] = self.mdp_reward.reward_perception
        self.extras["reward_cmd"] = self.mdp_reward.reward_cmd
        self.extras["reward_collision"] = self.mdp_reward.reward_collision
        self.extras["reward_guidance"] = self.mdp_reward.reward_guidance
        self.extras["reward_waypoint"] = self.mdp_reward.reward_waypoint
        self.extras["reward_timeout"] = self.mdp_reward.reward_timeout
        self.extras["reward_lin_vel"] = self.lin_vel_reward

    def _update_obs_dict(self):
        if self.obs_img_mode == "empty":
            self.obs_dict["obs"] = torch.cat(
                (
                    self.flat_drone_state,
                    self.flat_waypoint_info,
                    self.last_action,
                ),
                1,
            )
        elif self.obs_img_mode == "flat":
            self.obs_dict["obs"] = torch.cat(
                (
                    self.depth_image_batch.flatten(1),
                    self.flat_drone_state,
                    self.flat_waypoint_info,
                    self.last_action,
                ),
                1,
            )
        elif self.obs_img_mode == "dce":
            if self.enable_camera_sensors:
                self.obs_dict["obs"] = torch.cat(
                    (
                        self.dce.encode(self.depth_image_batch),
                        self.flat_drone_state,
                        self.flat_waypoint_info,
                        self.last_action,
                    ),
                    1,
                )
            else:
                self.obs_dict["obs"] = torch.cat(
                    (
                        self.dummy_encoded_img,
                        self.flat_drone_state,
                        self.flat_waypoint_info,
                        self.last_action,
                    ),
                    1,
                )

    def _update_extras(self):
        self.extras["crashed"] = self.crashed
        self.extras["finished"] = self.finished
        self.extras["time_outs"] = self.timeout_buf
        self.extras["progress"] = self.progress_buf
