# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: schema for base class configuration.

Used by Hydra. Defines template for base class YAML file.
"""

from dataclasses import dataclass


@dataclass
class Mode:
    export_scene: bool  # export scene to USD
    export_states: bool  # export states to NPY


@dataclass
class Sim:
    dt: float  # timestep size (default = 1.0 / 60.0)
    num_substeps: int  # number of substeps (default = 2)
    num_pos_iters: int  # number of position iterations for PhysX TGS solver (default = 4)
    num_vel_iters: int  # number of velocity iterations for PhysX TGS solver (default = 1)
    gravity_mag: float  # magnitude of gravitational acceleration
    add_damping: bool  # add damping to stabilize gripper-object interactions


@dataclass
class Env:
    env_spacing: float  # lateral offset between envs
    franka_depth: float  # depth offset of Franka base relative to env origin
    table_height: float  # height of table
    franka_friction: float  # coefficient of friction associated with Franka
    table_friction: float  # coefficient of friction associated with table


@dataclass
class FactorySchemaConfigBase:
    mode: Mode
    sim: Sim
    env: Env
