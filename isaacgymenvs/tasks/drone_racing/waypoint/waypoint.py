import math
from typing import List


class Waypoint:

    def __init__(
        self,
        index: int,
        xyz: List[float],
        rpy: List[float],
        length_y: float,
        length_z: float,
        gate: bool,
    ):
        self.index = index
        self.xyz = xyz
        self.rpy = rpy
        self.length_y = length_y
        self.length_z = length_z
        self.gate = gate

    def rpy_rad(self) -> List[float]:
        return [
            math.radians(self.rpy[0]),
            math.radians(self.rpy[1]),
            math.radians(self.rpy[2]),
        ]

    def compact_data(self) -> List[float]:
        gate = -1.0
        if self.gate:
            gate = 1.0
        data = self.xyz + self.rpy_rad() + [self.length_y, self.length_z, gate]
        return data
