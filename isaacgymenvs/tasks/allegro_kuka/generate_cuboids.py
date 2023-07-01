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
from os.path import join
from typing import Callable, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

FilterFunc = Callable[[List[int]], bool]


def generate_assets(
    scales, min_volume, max_volume, generated_assets_dir, base_mesh, base_cube_size_m, filter_funcs: List[FilterFunc]
):
    template_dir = join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/asset_templates")
    print(f"Assets template dir: {template_dir}")

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
    )
    template = env.get_template("cube_multicolor_allegro.urdf.template")  # <-- pass as function parameter?

    idx = 0
    for x_scale in scales:
        for y_scale in scales:
            for z_scale in scales:
                volume = x_scale * y_scale * z_scale / (100 * 100 * 100)
                if volume > max_volume:
                    continue
                if volume < min_volume:
                    continue

                curr_scales = [x_scale, y_scale, z_scale]
                curr_scales.sort()

                filtered = False
                for filter_func in filter_funcs:
                    if filter_func(curr_scales):
                        filtered = True

                if filtered:
                    continue

                asset = template.render(
                    base_mesh=base_mesh,
                    x_scale=base_cube_size_m * (x_scale / 100),
                    y_scale=base_cube_size_m * (y_scale / 100),
                    z_scale=base_cube_size_m * (z_scale / 100),
                )
                fname = f"{idx:03d}_cube_{x_scale}_{y_scale}_{z_scale}.urdf"
                idx += 1
                with open(join(generated_assets_dir, fname), "w") as fobj:
                    fobj.write(asset)


def filter_thin_plates(scales: List[int]) -> bool:
    """
    Skip cuboids where one dimension is much smaller than the other two - these are very hard to grasp.
    We return true if object needs to be skipped.
    """
    scales = sorted(scales)
    return scales[0] * 3 <= scales[1]


def generate_default_cube(assets_dir, base_mesh, base_cube_size_m):
    scales = [100]
    min_volume = max_volume = 1.0
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_cube_size_m, [])


def generate_small_cuboids(assets_dir, base_mesh, base_cube_size_m):
    scales = [100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300]
    min_volume = 1.0
    max_volume = 2.5
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_cube_size_m, [])


def generate_big_cuboids(assets_dir, base_mesh, base_cube_size_m):
    scales = [100, 125, 150, 200, 250, 300, 350]
    min_volume = 2.5
    max_volume = 15.0
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh, base_cube_size_m, [filter_thin_plates])


def filter_non_elongated(scales: List[int]) -> bool:
    """
    Skip cuboids that are not elongated. One dimension should be significantly larger than the other two.
    We return true if object needs to be skipped.
    """
    scales = sorted(scales)
    return scales[2] <= scales[0] * 3 or scales[2] <= scales[1] * 3


def generate_sticks(assets_dir, base_mesh, base_cube_size_m):
    scales = [100, 50, 75, 200, 300, 400, 500, 600]
    min_volume = 2.5
    max_volume = 6.0
    generate_assets(
        scales,
        min_volume,
        max_volume,
        assets_dir,
        base_mesh,
        base_cube_size_m,
        [filter_thin_plates, filter_non_elongated],
    )
