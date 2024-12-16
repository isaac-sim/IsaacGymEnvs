from isaacgym import gymapi, gymtorch
import torch

class SimpleRobotEnv:
    def __init__(self, gym, sim, env, num_envs):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.num_envs = num_envs

        # Example observation and reward buffers
        self.obs_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device='cpu')
        self.rew_buf = torch.zeros((num_envs,), dtype=torch.float32, device='cpu')

        # Target position (static for simplicity)
        self.target_pos = torch.tensor([0.5, 0.5, 0.5], device='cpu')

        # Create actors and get handles
        self.actor_handles = []
        for i in range(num_envs):
            actor_handle = self.gym.create_actor(self.env, self._create_robot_asset(), gymapi.Transform(), f"robot_{i}")
            self.actor_handles.append(actor_handle)

        # Get joint and link handles
        self.dof_states = self.gym.acquire_dof_state_tensor(sim)
        self.dof_pos = gymtorch.wrap_tensor(self.dof_states)[:, 0].view(num_envs, -1)
        self.dof_vel = gymtorch.wrap_tensor(self.dof_states)[:, 1].view(num_envs, -1)
        self.ee_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device='cpu')

        # Action scaling
        self.action_scale = 0.1

    def _create_robot_asset(self):
        """Load a simple robot asset."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_root = "./assets"
        asset_file = "urdf/xhand/xhand_right.urdf"
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise Exception("Failed to load robot asset")
        return asset

    def compute_observations(self):
        """
        Computes observations, concatenating joint positions and target position.
        """
        self.obs_buf = torch.cat([self.dof_pos, self.target_pos.expand(self.num_envs, -1)], dim=-1)

    def compute_rewards(self):
        """
        Computes rewards based on distance between the end-effector and the target.
        """
        dist_to_target = torch.norm(self.ee_pos - self.target_pos, dim=-1)
        self.rew_buf = -dist_to_target  # Negative reward for distance

        # Add a success reward for being close to the target
        success_mask = dist_to_target < 0.1  # Success threshold
        self.rew_buf[success_mask] += 10.0  # Bonus for success

    def step(self, actions):
        """
        Simulate one step in the environment.
        """
        # Apply actions to update joint positions
        actions = torch.clamp(actions, -1.0, 1.0) * self.action_scale
        self.dof_pos += actions
        
        # make sure tensor is contiguous
        self.dof_pos = self.dof_pos.contiguous()

        # Update the simulation
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Compute end-effector positions
        for i, actor_handle in enumerate(self.actor_handles):
            ee_link = self.gym.find_actor_rigid_body_handle(self.env, actor_handle, "right_hand_mid_tip") # Assuming :panda_leftfinger is the end-effector link
            ee_transform = self.gym.get_rigid_transform(self.env, ee_link)  
            self.ee_pos[i, :] = torch.tensor([ee_transform.p.x, ee_transform.p.y, ee_transform.p.z], device='cpu')

        # Compute observations and rewards
        self.compute_observations()
        self.compute_rewards()

# Example usage
gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX)
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

num_envs = 4
robot_env = SimpleRobotEnv(gym, sim, env, num_envs)

for step in range(10):  # Example of 10 simulation steps
    # Random actions for demonstration purposes
    actions = torch.rand((num_envs, 12), device='cpu') * 2 - 1
    robot_env.step(actions)
    print(f"Step {step + 1}:")
    print(f"Observations:\n{robot_env.obs_buf}")
    print(f"Rewards:\n{robot_env.rew_buf}")
