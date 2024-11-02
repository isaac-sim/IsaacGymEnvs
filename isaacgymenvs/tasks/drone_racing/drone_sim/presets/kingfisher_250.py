import math

import numpy as np

from isaacgym import gymapi
from ..controllers import SimpleBetaflightParams
from ..models import (
    BodyDragPolyParams,
    PropellerPolyParams,
    RotorPolyLagParams,
    WrenchSumParams,
)
from ...assets import DroneQuadcopterOptions


class Kingfisher250:
    """
    A collection of options, params, properties for the Kingfisher quad running modules at 250Hz.
    """

    def __init__(self, num_envs: int, device: str):
        # geometry
        self.arm_length = 0.125
        self.arm_angle = math.radians(102)
        self.num_rotors = 4
        x = self.arm_length * math.cos(self.arm_angle / 2)
        y = self.arm_length * math.sin(self.arm_angle / 2)
        self.rotor_x = [-x, x, -x, x]
        self.rotor_y = [y, y, -y, -y]
        self.rotor_dir = [1, -1, -1, 1]

        # sim
        self.dt = 1 / 250
        self.num_envs = num_envs
        self.device = device

        # module params
        self.quad_asset_options = DroneQuadcopterOptions()
        self.init_quad_asset_options()

        self.simple_bf_params = SimpleBetaflightParams()
        self.init_simple_bf_params()

        self.rotor_params = RotorPolyLagParams()
        self.init_rotor_params()

        self.propeller_params = PropellerPolyParams()
        self.init_propeller_params()

        self.body_drag_params = BodyDragPolyParams()
        self.init_body_drag_params()

        self.wrench_sum_params = WrenchSumParams()
        self.init_wrench_sum_params()

        self.camera_props = gymapi.CameraProperties()
        self.camera_pose = gymapi.Transform()
        self.init_camera_props()

    def init_quad_asset_options(self):
        self.quad_asset_options.file_name = "kingfisher_250"
        self.quad_asset_options.arm_length_front = self.arm_length
        self.quad_asset_options.arm_length_back = self.arm_length
        self.quad_asset_options.arm_thickness = 0.01
        self.quad_asset_options.arm_front_angle = self.arm_angle
        self.quad_asset_options.motor_diameter = 0.023
        self.quad_asset_options.motor_height = 0.006
        self.quad_asset_options.central_body_pos = [0.0, 0.0, 0.015]
        self.quad_asset_options.central_body_dim = [0.15, 0.05, 0.05]
        self.quad_asset_options.propeller_diameter = 0.12954
        self.quad_asset_options.propeller_height = 0.01
        self.quad_asset_options.mass = 0.752
        self.quad_asset_options.center_of_mass = [0.0, 0.0, 0.0]
        self.quad_asset_options.diagonal_inertia = [0.0025, 0.0021, 0.0043]
        self.quad_asset_options.principle_axes_q = [1.0, 0.0, 0.0, 0.0]
        self.quad_asset_options.asset_options = gymapi.AssetOptions()

    def init_simple_bf_params(self):
        # basic settings
        self.simple_bf_params.num_envs = self.num_envs
        self.simple_bf_params.device = self.device
        self.simple_bf_params.dt = self.dt
        # rate mapping
        self.simple_bf_params.center_sensitivity = [100.0, 100.0, 100.0]
        self.simple_bf_params.max_rate = [670.0, 670.0, 670.0]
        self.simple_bf_params.rate_expo = [0.0, 0.0, 0.0]
        # pid
        self.simple_bf_params.kp = [70.0, 70.0, 125.0]
        self.simple_bf_params.ki = [0.5, 0.5, 25.0]
        self.simple_bf_params.kd = [1.0, 1.0, 0.0]
        self.simple_bf_params.kff = [0.0, 0.0, 0.0]
        self.simple_bf_params.iterm_lim = [5.0, 5.0, 5.0]
        self.simple_bf_params.pid_sum_lim = [1000.0, 1000.0, 1000.0]
        self.simple_bf_params.dterm_lpf_cutoff = 1000
        # mixer
        self.simple_bf_params.rotors_x = self.rotor_x
        self.simple_bf_params.rotors_y = self.rotor_y
        self.simple_bf_params.rotors_dir = self.rotor_dir
        self.simple_bf_params.pid_sum_mixer_scale = 1000.0
        self.simple_bf_params.output_idle = 0.05
        self.simple_bf_params.throttle_boost_gain = 0.0
        self.simple_bf_params.throttle_boost_freq = 125.0
        self.simple_bf_params.thrust_linearization_gain = 0.4

    def init_rotor_params(self):
        self.rotor_params.num_envs = self.num_envs
        self.rotor_params.device = self.device
        self.rotor_params.dt = self.dt
        self.rotor_params.num_rotors = self.num_rotors
        self.rotor_params.rotors_dir = self.rotor_dir
        self.rotor_params.spinup_time_constant = 0.033
        self.rotor_params.slowdown_time_constant = 0.033
        self.rotor_params.k_rpm_quadratic = -13421.95
        self.rotor_params.k_rpm_linear = 37877.42
        self.rotor_params.rotor_diagonal_inertia = [0.0, 0.0, 9.3575e-6]
        self.rotor_params.rotor_principle_axes_q = [1.0, 0.0, 0.0, 0.0]

    def init_propeller_params(self):
        self.propeller_params.num_envs = self.num_envs
        self.propeller_params.device = self.device
        self.propeller_params.num_props = self.num_rotors
        self.propeller_params.prop_dir = self.rotor_dir
        self.propeller_params.k_force_quadratic = 2.1549e-08
        self.propeller_params.k_force_linear = -4.5101e-05
        self.propeller_params.k_torque_quadratic = 2.1549e-08 * 0.022
        self.propeller_params.k_torque_linear = -4.5101e-05 * 0.022

    def init_body_drag_params(self):
        # basic settings
        self.body_drag_params.num_envs = self.num_envs
        self.body_drag_params.device = self.device
        self.body_drag_params.air_density = 1.204
        # translational
        self.body_drag_params.a_trans = [1.5e-2, 1.5e-2, 3.0e-2]
        self.body_drag_params.k_trans_quadratic = [1.04, 1.04, 1.04]
        self.body_drag_params.k_trans_linear = [0.0, 0.0, 0.0]
        # rotational
        self.body_drag_params.a_rot = [1e-2, 1e-2, 1e-2]
        self.body_drag_params.k_rot_quadratic = [0.0, 0.0, 0.0]
        self.body_drag_params.k_rot_linear = [0.0, 0.0, 0.0]

    def init_wrench_sum_params(self):
        self.wrench_sum_params.num_envs = self.num_envs
        self.wrench_sum_params.device = self.device
        self.wrench_sum_params.num_positions = self.num_rotors
        self.wrench_sum_params.position_x = self.rotor_x
        self.wrench_sum_params.position_y = self.rotor_y
        self.wrench_sum_params.position_z = [0.0, 0.0, 0.0, 0.0]

    def init_camera_props(self):
        self.camera_props.enable_tensors = True
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.horizontal_fov = 90

        self.camera_pose.p = gymapi.Vec3(0.08, 0.0, 0.015)
        self.camera_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), np.radians(-20.0)
        )
