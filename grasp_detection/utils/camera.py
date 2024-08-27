from dataclasses import dataclass
from typing import Type

import numpy as np
from PIL import Image

@dataclass
class CameraParameters:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    colors: np.ndarray
    depths: np.ndarray
    masks: np.ndarray