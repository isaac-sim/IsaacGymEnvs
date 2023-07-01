import random
from typing import List

# Versioning -- you can change this number and keep a changelog below to keep track of your experiments as you go.
version = "v1"


def seeds(num_seeds) -> List[int]:
    return [random.randrange(1000000, 9999999) for _ in range(num_seeds)]


default_num_frames: int = 10_000_000_000
