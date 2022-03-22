Release Notes
=============

1.2.0
-----

Added AMP (Adversarial Motion Priors) training environment.

Minor changes in base VecTask class.

1.1.0
-----

Added Anymal Rough Terrain and Trifinger training environments.

Added `self.timeout_buf` that stores the information if the reset happened because of the episode reached to the maximum length or because of some other termination conditions. Is stored in extra info: `self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)`.  Updated PPO configs to use this information during training with `value_bootstrap: True`.

1.0.0
-----
Initial release