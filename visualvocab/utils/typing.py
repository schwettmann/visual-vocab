"""Some useful type aliases relevant to this project."""
import pathlib
from typing import List, Tuple, Union

import torch

Device = Union[str, torch.device]
Layer = Union[int, str]
PathLike = Union[str, pathlib.Path]

StrSequence = Union[List[str], Tuple[str, ...]]
