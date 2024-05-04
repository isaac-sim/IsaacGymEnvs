# Copyright (c) 2018-2023, NVIDIA Corporation
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

import os
from pathlib import Path

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

import torch

# load in XML mjcf file and save zero rotation pose in npy format
xml_path = Path(os.environ.get("MODEL_DIR")) / "robot.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)
print(skeleton)

zero_pose = SkeletonState.zero_pose(skeleton)
zero_pose.to_file("data/nv_humanoid.npy")

local_rotation = zero_pose.local_rotation.clone()
local_rotation[0] = torch.tensor([1, 0, 0, 1])
local_rotation[1] = torch.tensor([0, 2.32051e-08, -1, 6.96153e-08])
local_rotation[2] = torch.tensor([-0.672498, -0.672499, 0.218508, 0.218508])
local_rotation[3] = torch.tensor([0.379928, -0.596368, 0.596368, -0.379928])
local_rotation[4] = torch.tensor([0.707107, 0.707107, 0, 0])
local_rotation[5] = torch.tensor([0.5, -0.5, -0.5, -0.5])
local_rotation[6] = torch.tensor([-0.612372, -0.353553, 0.612372, -0.353553])
local_rotation[7] = torch.tensor([0.5, -0.5, 0.5, -0.5])
local_rotation[8] = torch.tensor([-0.706138, -0.706138, 0.0370071, 0.0370071])
local_rotation[9] = torch.tensor([-0.218508, -0.218508, -0.672498, -0.6724991])
local_rotation[10] = torch.tensor([0.218508, -0.218508, -0.672499, 0.672498])
local_rotation[11] = torch.tensor([-0.32102, -0.32102, 0.630037, 0.630037])
local_rotation[12] = torch.tensor([0.379928, -0.596368, -0.596368, 0.379928])
local_rotation[13] = torch.tensor([0.707107, 0.707107, 0, 0])
local_rotation[14] = torch.tensor([0.5, -0.5, -0.5, -0.5])
local_rotation[15] = torch.tensor([0.612372, -0.353553, -0.612372, -0.353553])
local_rotation[16] = torch.tensor([0.5, -0.5, 0.5, -0.5])
local_rotation[17] = torch.tensor([-0.706138, -0.706138, 0.0370071, 0.0370071])
local_rotation[18] = torch.tensor([-0.218508, -0.218508, -0.672498, -0.672499])
local_rotation[19] = torch.tensor([0.218508, -0.218508, -0.672499, 0.672498])
local_rotation[20] = torch.tensor([-0.32102, -0.32102, 0.630037, 0.630037])
local_rotation[21] = torch.tensor([-1, -4.64102e-08, 0, 0])
local_rotation[22] = torch.tensor([-0.5, 0.5, 0.5, 0.5])
local_rotation[23] = torch.tensor([0.353553, 0.353553, 0.612372, 0.612372])
local_rotation[24] = torch.tensor([-0.183013, -0.683013, -0.183013, 0.683013])
local_rotation[25] = torch.tensor([3.44263e-08, -0.258819, -0.965926, -6.00592e-09])
local_rotation[26] = torch.tensor([-0.866025, -4.01924e-08, 0, -0.5])
local_rotation[27] = torch.tensor([0.5, -8.49366e-09, -8.49366e-09, -0.866025])
local_rotation[28] = torch.tensor([0.707107, 0, 0, 0.707107])
local_rotation[29] = torch.tensor([0.707107, 3.53553e-08, 0.707107, 1.64085e-08])
local_rotation[30] = torch.tensor([-0.5, 0.5, -0.5, -0.5])
local_rotation[31] = torch.tensor([-0.353553, -0.353553, 0.612372, 0.612372])
local_rotation[32] = torch.tensor([-0.683013, 0.183013, -0.683013, -0.183013])
local_rotation[33] = torch.tensor([3.44263e-08, -0.258819, 0.965926, 6.00592e-09])
local_rotation[34] = torch.tensor([-0.866025, -4.01924e-08, 0, 0.5])
local_rotation[35] = torch.tensor([0.707107, 0, 0, -0.707107])
local_rotation[36] = torch.tensor([1.89469e-08, -0.707107, 0, -0.707107])
local_rotation[37] = torch.tensor([0.5, -8.49366e-09, 8.49366e-09, 0.866025])

new_pose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=skeleton,
             r=local_rotation,
             t=zero_pose.root_translation,
             is_local=False
          )
# # visualize zero rotation pose
plot_skeleton_state(new_pose)