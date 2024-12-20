import os
import time

import numpy as np
import torch
import torch.distributed as dist
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import print_statistics, swap_and_flatten01
from typing import Optional
from tqdm import tqdm


class DRAgent(A2CAgent):

    def __init__(self, base_name, params):
        print("+++ DRAgent")

        self.dones: Optional[torch.Tensor] = None
        self.current_rewards: Optional[torch.Tensor] = None
        self.current_shaped_rewards: Optional[torch.Tensor] = None
        self.current_lengths: Optional[torch.Tensor] = None
        self.current_rewards_progress: Optional[torch.Tensor] = None
        self.current_rewards_perception: Optional[torch.Tensor] = None
        self.current_rewards_cmd: Optional[torch.Tensor] = None
        self.current_rewards_collision: Optional[torch.Tensor] = None
        self.current_rewards_guidance: Optional[torch.Tensor] = None
        self.current_rewards_waypoint: Optional[torch.Tensor] = None
        self.current_rewards_timeout: Optional[torch.Tensor] = None
        self.current_rewards_lin_vel: Optional[torch.Tensor] = None

        super().__init__(base_name, params)

        # we want to track separate reward terms
        self.game_rewards_progress = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_perception = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_cmd = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_collision = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_guidance = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_waypoint = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_timeout = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)
        self.game_rewards_lin_vel = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.ppo_device)

        print(self.model)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        # NOTE: reset before every rollout
        # self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("+++ broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            # Note: reset before every rollout
            # TODO: enable from cfg?
            print("DRAgent: resetting envs")
            self.obs = self.env_reset()
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = (
                    self.curr_frames * self.world_size
                    if self.multi_gpu
                    else self.curr_frames
                )
                self.frame += curr_frames

                print_statistics(
                    self.print_stats,
                    curr_frames,
                    step_time,
                    scaled_play_time,
                    scaled_time,
                    epoch_num,
                    self.max_epochs,
                    frame,
                    self.max_frames,
                )

                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )

                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        "losses/bounds_loss",
                        torch_ext.mean_list(b_losses).item(),
                        frame,
                    )

                if self.has_soft_aug:
                    raise NotImplementedError

                mean_rewards = None
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            rewards_name + "/step".format(i), mean_rewards[i], frame
                        )
                        self.writer.add_scalar(
                            rewards_name + "/iter".format(i), mean_rewards[i], epoch_num
                        )
                        self.writer.add_scalar(
                            rewards_name + "/time".format(i),
                            mean_rewards[i],
                            total_time,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/step".format(i),
                            mean_shaped_rewards[i],
                            frame,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/iter".format(i),
                            mean_shaped_rewards[i],
                            epoch_num,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/time".format(i),
                            mean_shaped_rewards[i],
                            total_time,
                        )

                    # NOTE: add more entries to writer
                    # TODO: better place to put all these?
                    self.writer.add_scalar(
                        "rewards/progress/step",
                        self.game_rewards_progress.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/perception/step",
                        self.game_rewards_perception.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/cmd/step",
                        self.game_rewards_cmd.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/collision/step",
                        self.game_rewards_collision.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/guidance/step",
                        self.game_rewards_guidance.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/waypoint/step",
                        self.game_rewards_waypoint.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/collision/step",
                        self.game_rewards_collision.get_mean()[0],
                        frame,
                    )
                    self.writer.add_scalar(
                        "rewards/lin_vel/step",
                        self.game_rewards_lin_vel.get_mean()[0],
                        frame,
                    )

                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar(
                        "episode_lengths/iter", mean_lengths, epoch_num
                    )
                    self.writer.add_scalar(
                        "episode_lengths/time", mean_lengths, total_time
                    )

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = (
                        self.config["name"]
                        + "_ep_"
                        + str(epoch_num)
                        + "_rew_"
                        + str(mean_rewards[0])
                    )

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(
                                os.path.join(self.nn_dir, "last_" + checkpoint_name)
                            )

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.save_best_after
                    ):
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config["name"]))

                        if "score_to_win" in self.config:
                            if self.last_mean_rewards > self.config["score_to_win"]:
                                print("Maximum reward achieved. Network won!")
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max epochs reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_ep_"
                            + str(epoch_num)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if self.frame >= self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max frames reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_frame_"
                            + str(self.frame)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num

    def init_tensors(self):
        super().init_tensors()
        self.current_rewards_progress = torch.zeros_like(self.current_rewards)
        self.current_rewards_perception = torch.zeros_like(self.current_rewards)
        self.current_rewards_cmd = torch.zeros_like(self.current_rewards)
        self.current_rewards_collision = torch.zeros_like(self.current_rewards)
        self.current_rewards_guidance = torch.zeros_like(self.current_rewards)
        self.current_rewards_waypoint = torch.zeros_like(self.current_rewards)
        self.current_rewards_timeout = torch.zeros_like(self.current_rewards)
        self.current_rewards_lin_vel = torch.zeros_like(self.current_rewards)

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in tqdm(range(self.horizon_length)):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.time()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_rewards_progress += infos["reward_progress"].unsqueeze(1)
            self.current_rewards_perception += infos["reward_perception"].unsqueeze(1)
            self.current_rewards_cmd += infos["reward_cmd"].unsqueeze(1)
            self.current_rewards_collision += infos["reward_collision"].unsqueeze(1)
            self.current_rewards_guidance += infos["reward_guidance"].unsqueeze(1)
            self.current_rewards_waypoint += infos["reward_waypoint"].unsqueeze(1)
            self.current_rewards_timeout += infos["reward_timeout"].unsqueeze(1)
            self.current_rewards_lin_vel += infos["reward_lin_vel"].unsqueeze(1)
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_rewards_progress.update(
                self.current_rewards_progress[env_done_indices]
            )
            self.game_rewards_perception.update(
                self.current_rewards_perception[env_done_indices]
            )
            self.game_rewards_cmd.update(self.current_rewards_cmd[env_done_indices])
            self.game_rewards_collision.update(
                self.current_rewards_collision[env_done_indices]
            )
            self.game_rewards_guidance.update(
                self.current_rewards_guidance[env_done_indices]
            )
            self.game_rewards_waypoint.update(
                self.current_rewards_waypoint[env_done_indices]
            )
            self.game_rewards_timeout.update(
                self.current_rewards_timeout[env_done_indices]
            )
            self.game_rewards_lin_vel.update(
                self.current_rewards_lin_vel[env_done_indices]
            )
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()
            not_dones_un_sq = not_dones.unsqueeze(1)

            self.current_rewards = self.current_rewards * not_dones_un_sq
            self.current_rewards_progress = (
                self.current_rewards_progress * not_dones_un_sq
            )
            self.current_rewards_perception = (
                self.current_rewards_perception * not_dones_un_sq
            )
            self.current_rewards_cmd = self.current_rewards_cmd * not_dones_un_sq
            self.current_rewards_collision = (
                self.current_rewards_collision * not_dones_un_sq
            )
            self.current_rewards_guidance = (
                self.current_rewards_guidance * not_dones_un_sq
            )
            self.current_rewards_waypoint = (
                self.current_rewards_waypoint * not_dones_un_sq
            )
            self.current_rewards_timeout = (
                self.current_rewards_timeout * not_dones_un_sq
            )
            self.current_rewards_lin_vel = (
                self.current_rewards_lin_vel * not_dones_un_sq
            )
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones_un_sq
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        return batch_dict

    def play_steps_rnn(self):
        raise NotImplementedError("Not tested")
