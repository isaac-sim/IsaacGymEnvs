from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from ..utils import FirstOrderLowPassFilterParams, FirstOrderLowPassFilter


@dataclass
class SimpleBetaflightParams:
    # number of envs in parallel
    num_envs: int = 64

    # tensor device
    device: str = "cuda"

    # control period (seconds)
    dt: float = 1 / 500

    # stick position to angular rates (deg/s), for RPY
    # the pilot can adjust their Rates to suit their flying style
    # racers prefer a more linear curve with a maximum turn rate of around 550-650 deg/s
    # freestyle typically uses a combination of a soft center region with high maximum turn rates (850-1200 deg/s)
    # cinematic flying will be smoother with a flatter center region
    center_sensitivity: List[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0]
    )
    max_rate: List[float] = field(default_factory=lambda: [670.0, 670.0, 670.0])
    rate_expo: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # PID (rad) for RPY
    kp: List[float] = field(default_factory=lambda: [150.0, 150.0, 100.0])
    ki: List[float] = field(default_factory=lambda: [2.0, 2.0, 15.0])
    kd: List[float] = field(default_factory=lambda: [2.0, 2.0, 0.0])
    kff: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    iterm_lim: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    pid_sum_lim: List[float] = field(default_factory=lambda: [1000.0, 1000.0, 1000.0])

    # d-term low pass filter cutoff frequency in Hz
    dterm_lpf_cutoff: float = 200

    # rotor positions in body FRD frame
    # all rotors are assumed to only produce thrust along the body-z axis
    # so z component does not matter anyway
    # rotor indexing: https://betaflight.com/docs/wiki/configurator/motors-tab
    rotors_x: List[float] = field(
        default_factory=lambda: [-0.078665, 0.078665, -0.078665, 0.078665]
    )
    rotors_y: List[float] = field(
        default_factory=lambda: [0.097143, 0.097143, -0.097143, -0.097143]
    )
    rotors_dir: List[int] = field(default_factory=lambda: [1, -1, -1, 1])
    pid_sum_mixer_scale: float = 1000.0

    # output idle
    output_idle: float = 0.05

    # throttle boost
    throttle_boost_gain: float = 10.0
    throttle_boost_freq: float = 50.0

    # thrust linearization
    thrust_linearization_gain: float = 0.4


class SimpleBetaflight:
    """
    Simplified Betaflight rate PID control.

    I/O:
        - In: normalized stick positions in AETR channels, from -1 to 1.
        - Out: normalized rotor command of the rotors (u from 0 to 1).

    Implemented:
        - Actual rates mapping from AETR to angular velocity.
        - Angular rate PID with error-derivative I-term, D-term LPF, and FF based on setpoint value.
        - Mixing supporting customizable airframe.
        - AirMode using betaflight default (LEGACY) mixer adjustment.
        - Throttle Boost: throttle command is boosted by high-frequency component of itself.
        - Thrust Linearization: boosting output at low throttle, and lowering it at high throttle.

    Not implemented:
        - Antigravity: boosting PI during fast throttle movement.
        - Throttle PID Attenuation: reducing PID at high throttle to cope with motor noise.
        - I-term relax: disabling I-term calculation during fast maneuvers.
        - Dynamic damping: higher D-term coefficient during fast maneuvers.
        - Integrated yaw: integrating PID sum about z-axis before putting it into the mixer.
        - Absolute control: for better tracking to sticks, particularly during rotations involving fast yaw movement.
        - Sensor noise (gyro noise) and additional filtering (gyro filters, notch filters).
        - Dynamic Idle: controlling the minimum command level using PID to prevent motor-ESC de-synchronization.
        - Battery voltage compensation: for consistent response throughout a battery run.

    Reference:
        - [1] https://betaflight.com/docs/wiki
        - [2] https://www.desmos.com/calculator/r5pkxlxhtb
        - [3] https://en.wikipedia.org/wiki/Low-pass_filter
    """

    def __init__(self, params: SimpleBetaflightParams):
        self.params = params
        self.all_env_id = torch.arange(params.num_envs, device=params.device)

        # input
        self.command = torch.zeros(params.num_envs, 4, device=params.device)

        # rate
        self.center_sensitivity = torch.tensor(
            params.center_sensitivity, device=params.device
        )
        self.max_rate = torch.tensor(params.max_rate, device=params.device)
        self.rate_expo = torch.tensor(params.rate_expo, device=params.device)

        # pid
        self.kp = torch.tensor(params.kp, device=params.device)
        self.ki = torch.tensor(params.ki, device=params.device)
        self.kd = torch.tensor(params.kd, device=params.device)
        self.kff = torch.tensor(params.kff, device=params.device)
        self.iterm_lim = torch.tensor(params.iterm_lim, device=params.device)
        self.pid_sum_lim = torch.tensor(params.pid_sum_lim, device=params.device)
        self.int_err_ang_vel = torch.zeros(params.num_envs, 3, device=params.device)
        self.last_ang_vel = torch.zeros(params.num_envs, 3, device=params.device)
        dterm_lpf_params = FirstOrderLowPassFilterParams()
        dterm_lpf_params.device = params.device
        dterm_lpf_params.dim = self.last_ang_vel.size()
        dterm_lpf_params.dt = params.dt
        dterm_lpf_params.cutoff_frequency = params.dterm_lpf_cutoff
        dterm_lpf_params.initial_value = 0.0
        self.dterm_lpf = FirstOrderLowPassFilter(dterm_lpf_params)

        # mixing table
        if not (
            len(params.rotors_x) == len(params.rotors_y)
            and len(params.rotors_y) == len(params.rotors_dir)
        ):
            raise ValueError("Rotors configuration error.")
        self.num_rotors = len(params.rotors_x)
        rotors_x_abs = [abs(item) for item in params.rotors_x]
        rotors_y_abs = [abs(item) for item in params.rotors_y]
        scale = max(max(rotors_x_abs), max(rotors_y_abs))
        mix_table_data = []
        for i in range(self.num_rotors):
            mix_table_data.append(
                [
                    1,  # throttle
                    -params.rotors_y[i] / scale,  # roll
                    params.rotors_x[i] / scale,  # pitch
                    -params.rotors_dir[i],  # yaw
                ]
            )
        self.mix_table = torch.tensor(mix_table_data, device=params.device)

        # throttle boost
        throttle_boost_lpf_params = FirstOrderLowPassFilterParams()
        throttle_boost_lpf_params.device = params.device
        throttle_boost_lpf_params.dim = torch.Size([params.num_envs])
        throttle_boost_lpf_params.dt = params.dt
        throttle_boost_lpf_params.cutoff_frequency = params.throttle_boost_freq
        throttle_boost_lpf_params.initial_value = 0.0
        self.throttle_boost_lpf = FirstOrderLowPassFilter(throttle_boost_lpf_params)

        # thrust linearization
        self.thrust_linearization_throttle_compensation = (
            params.thrust_linearization_gain - 0.5 * params.thrust_linearization_gain**2
        )

    def reset(self, env_id: torch.Tensor = None):
        if env_id is None:
            env_id = self.all_env_id

        self.int_err_ang_vel[env_id, ...] = 0
        self.last_ang_vel[env_id, ...] = 0
        self.dterm_lpf.reset(env_id)
        self.throttle_boost_lpf.reset(env_id)

    def set_command(self, command: torch.Tensor):
        """
        Sets the command (stick positions).

        Args:
            command: normalized stick positions in tensor shaped (num_envs, 4).
        """

        self.command[:] = command

    def compute(self, ang_vel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs main controller logic.

        Args:
           ang_vel: the current sensed angular velocity in rad/s, shaped (num_envs, 3).

        Returns:
           - Desired angular velocity in rad/s, shaped (num_envs, 3).
           - Normalized rotor command u[0, 1], shaped (num_envs, num_rotors).
        """

        # desired angular velocity
        des_ang_vel = _compute_input_map_script(
            command=self.command,
            center_sensitivity=self.center_sensitivity,
            max_rate=self.max_rate,
            rate_expo=self.rate_expo,
        )

        # angular velocity error
        err_ang_vel = des_ang_vel - ang_vel

        # integral of error rate, and limit the integral amount
        self.int_err_ang_vel += err_ang_vel
        self.int_err_ang_vel.clamp_(min=-self.iterm_lim, max=self.iterm_lim)

        # derivative term
        d_ang_vel = self.dterm_lpf.get_output()

        pid_sum = _compute_pid_sum_script(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            kff=self.kff,
            pid_sum_lim=self.pid_sum_lim,
            pid_sum_mixer_scale=self.params.pid_sum_mixer_scale,
            err_ang_vel=err_ang_vel,
            int_err_ang_vel=self.int_err_ang_vel,
            d_ang_vel=d_ang_vel,
            des_ang_vel=des_ang_vel,
        )

        # update dterm low-pass filter
        self.dterm_lpf.update((ang_vel - self.last_ang_vel) / self.params.dt)
        self.last_ang_vel[:] = ang_vel

        # mixing
        cmd_t = (self.command[:, 2] + 1) / 2  # (num_envs, )
        throttle_low_freq_component = self.throttle_boost_lpf.get_output()

        u = _compute_mixing_script(
            mix_table=self.mix_table,
            throttle_boost_gain=self.params.throttle_boost_gain,
            thrust_linearization_throttle_compensation=self.thrust_linearization_throttle_compensation,
            thrust_linearization_gain=self.params.thrust_linearization_gain,
            output_idle=self.params.output_idle,
            pid_sum=pid_sum,
            cmd_t=cmd_t,
            throttle_low_freq_component=throttle_low_freq_component,
        )

        self.throttle_boost_lpf.update(cmd_t)

        # return results
        return des_ang_vel, u


@torch.jit.script
def _compute_input_map_script(
    command: torch.Tensor,
    center_sensitivity: torch.Tensor,
    max_rate: torch.Tensor,
    rate_expo: torch.Tensor,
) -> torch.Tensor:
    """
    Maps stick positions to desired body angular velocity:
    https://betaflight.com/docs/wiki/guides/current/Rate-Calculator.

    Assuming FRD body frame:
    channel A -> roll (body x),
    channel E -> pitch (body y),
    channel R -> yaw (body z).

    Let x[-1, 1] be the stick position, d the center sensitivity, f the max rate, g the expo,
    desired body rate = sgn(x) * ( d|x| + (f-d) * ( (1-g)x^2 + gx^6 ) )
    """

    cmd_aer = command[:, [0, 1, 3]]
    des_body_rates = torch.sgn(cmd_aer) * (
        center_sensitivity * torch.abs(cmd_aer)
        + (max_rate - center_sensitivity)
        * ((1 - rate_expo) * torch.pow(cmd_aer, 2) + rate_expo * torch.pow(cmd_aer, 6))
    )
    return torch.deg2rad(des_body_rates)


@torch.jit.script
def _compute_pid_sum_script(
    kp: torch.Tensor,
    ki: torch.Tensor,
    kd: torch.Tensor,
    kff: torch.Tensor,
    pid_sum_lim: torch.Tensor,
    pid_sum_mixer_scale: float,
    err_ang_vel: torch.Tensor,
    int_err_ang_vel: torch.Tensor,
    d_ang_vel: torch.Tensor,
    des_ang_vel: torch.Tensor,
) -> torch.Tensor:
    # PID sum and clamp
    pid_sum = (
        kp * err_ang_vel + ki * int_err_ang_vel - kd * d_ang_vel + kff * des_ang_vel
    )
    pid_sum.clamp_(min=-pid_sum_lim, max=pid_sum_lim)

    # scale the PID sum before mixing
    pid_sum /= pid_sum_mixer_scale

    return pid_sum


@torch.jit.script
def _compute_mixing_script(
    mix_table: torch.Tensor,
    throttle_boost_gain: float,
    thrust_linearization_throttle_compensation: float,
    thrust_linearization_gain: float,
    output_idle: float,
    pid_sum: torch.Tensor,
    cmd_t: torch.Tensor,
    throttle_low_freq_component: torch.Tensor,
):
    # find desired motor command from RPY PID, shape (num_envs, num_rotors)
    rpy_u = torch.matmul(mix_table[:, 1:], pid_sum.T).T

    # u range for each environment, shape (num_envs, )
    rpy_u_max = torch.max(rpy_u, 1).values
    rpy_u_min = torch.min(rpy_u, 1).values
    rpy_u_range = rpy_u_max - rpy_u_min

    # normalization factor
    norm_factor = 1 / rpy_u_range  # (num_envs, )
    norm_factor.clamp_(max=1.0)

    # mixer adjustment
    rpy_u_normalized = norm_factor.view(-1, 1) * rpy_u
    rpy_u_normalized_max = norm_factor * rpy_u_max
    rpy_u_normalized_min = norm_factor * rpy_u_min

    # throttle boost
    throttle_high_freq_component = cmd_t - throttle_low_freq_component
    throttle = cmd_t + throttle_boost_gain * throttle_high_freq_component
    throttle.clamp_(min=0.0, max=1.0)

    # thrust linearization step 1
    throttle /= 1 + thrust_linearization_throttle_compensation * torch.pow(
        1 - throttle, 2
    )

    # constrain throttle so it won't clip any outputs
    throttle.clamp_(min=-rpy_u_normalized_min, max=(1 - rpy_u_normalized_max))

    # synthesize output
    u_rpy_t = rpy_u_normalized + throttle.view(-1, 1)

    # thrust linearization step 2
    u_rpy_t *= 1 + thrust_linearization_gain * torch.pow(1 - u_rpy_t, 2)

    # calculate final u based on idle
    u = output_idle + (1 - output_idle) * u_rpy_t

    return u
