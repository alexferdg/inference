from typing import Dict, List, Tuple, Union

import torch
import os
from pathlib import Path
from jinja2 import Template

BASE_DIR_PATH: Path = Path(os.path.abspath(__file__)).parent.parent
WD_PATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_ID: str = "bert-base-uncased"
TORCH_DTYPE: torch.dtype = torch.float16 
MAX_POSITION_EMBEDDINGS: int = 1026
DEVICE: str = "cpu"