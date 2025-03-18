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
import math
import os
from os.path import join
from typing import Callable, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

FilterFunc = Callable[[List[int]], bool]


def generate_assets(
    scales, min_volume, max_volume, generated_assets_dir, base_mesh, base_sphere_size_m, filter_funcs: List[FilterFunc]
):
    template_dir = join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/asset_templates")
    print(f"Assets template dir: {template_dir}")

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
    )
    template = env.get_template("ball_allegro.urdf.template")  # <-- pass as function parameter?

    idx = 0
    for radius in scales:
        volume = 4/3 * math.pi * radius**3 / (100 * 100 * 100)
        if volume > max_volume:
            continue
        if volume < min_volume:
            continue

        curr_scales = [radius, radius, radius]

        filtered = False
        for filter_func in filter_funcs:
            if filter_func(curr_scales):
                filtered = True

        if filtered:
            continue

        asset = template.render(
            base_mesh=base_mesh,
            radius=base_sphere_size_m * (radius / 100)
        )
        fname = f"{idx:03d}_ball_{radius}_{radius}_{radius}.urdf"
        idx += 1
        with open(join(generated_assets_dir, fname), "w") as fobj:
            fobj.write(asset)

def generate_default_ball(assets_dir, base_mesh, base_ball_size_m):
    scales = [100]
    min_volume = max_volume = 1.0
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_ball_size_m, [])


def generate_small_balls(assets_dir, base_mesh, base_ball_size_m):
    scales = [100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300]
    min_volume = 1.0
    max_volume = 2.5
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_ball_size_m, [])


def generate_big_balls(assets_dir, base_mesh, base_ball_size_m):
    scales = [100, 125, 150, 200, 250, 300, 350]
    min_volume = 2.5
    max_volume = 15.0
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_ball_size_m, [])
