from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
    )

import numpy as np
import torch
from torch import mps
import random
from time import perf_counter

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from measures_util import (
    end_measure,
    log_measures,
    start_measure,
)
from typing import Dict, Iterable, List, Optional, Tuple, Union
import time
import os
import argparse
import random
from utils import get_logger

from globals_vars import (
    MODEL_ID,
    DEVICE,
    TORCH_DTYPE,
    MAX_POSITION_EMBEDDINGS,
)

LOGGER = get_logger("BERT model deployment", "info")
""""
Description: Return random integer in range [a, b], including both end points.

args: (None)
    Internal args:
        a: 0
        b: 1000000

Return:
    pseudo-random number given a seed
"""
def random_seed() -> int:
    current_time = time.time()
    random.seed(current_time)
    return random.randint(0,1000000)

# Reproducibility
def set_seed(seed: int, deterministic=True) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # Check that MPS is available
    if DEVICE == "mps":
        pass
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


class GenerationConfig(BaseModel):
    prompts: List[str] = [None]
    generate_kwargs: Optional[dict] = {}

parser = argparse.ArgumentParser("Custom device selection")
parser.add_argument("--device",
                    default="cpu",
                    type=str
                    )
args = parser.parse_args()
# device validation
if args.device == "mps" and torch.backends.mps.is_available():
    if torch.backends.mps.is_built():
        DEVICE = "mps"
if args.device == "cuda" and torch.cuda.is_available():
    DEVICE = "cuda"
elif args.device == "cpu" and not torch.backends.mps.is_available() and torch.cuda.is_available():
    DEVICE = "cpu"


app = FastAPI()
origins = ["*"]
# ToDo: Study security concerns
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def from_pretrained():
    globals()
    global pipe

    # Create causal language model generation pipeline
    LOGGER.info("Create BERT unmasking pipeline")
    LOGGER.info(f"Hardware Device: {DEVICE}")
    """
    References:
        - https://github.com/huggingface/transformers/blob/1fe1e3caa44617047f149bcc0c0b566343b714a7/src/transformers/pipelines/text_generation.py#L71
        - https://huggingface.co/docs/transformers/main_classes/pipelines
        - https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py
    
    Args:
        task: str = None,
        model: Optional = None,
        config: Optional[Union[str, PretrainedConfig]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
        feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
        image_processor: Optional[Union[str, BaseImageProcessor]] = None,
        framework: Optional[str] = None,
        revision: Optional[str] = None,
        use_fast: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        device: Optional[Union[int, str, "torch.device"]] = None,
        device_map=None,
        torch_dtype=None,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Dict[str, Any] = None,
        pipeline_class: Optional[Any] = None,
        **kwargs,
    ) -> Pipeline:
    """
    start_measures = start_measure()
    pipe = pipeline(
                task="fill-mask",
                model=MODEL_ID,
                tokenizer=MODEL_ID,
                device=DEVICE
    )
    end_measures = end_measure(start_measures)
    log_measures(end_measures, "Pipeline creation time")
    
    if DEVICE == "mps":
        LOGGER.info(f"Current Apple MPS dvice memory allocated: {mps.current_allocated_memory() / 10**9} GB VRAM")

@app.post("/generate")
async def generate(generation_config: GenerationConfig):
    globals()
    global pipe
    # HTTP request body validation
    body = generation_config.dict()
    batch: List[str] = body["prompts"] if isinstance(body["prompts"], list) else None
    if batch is None:
        return JSONResponse(status_code=200,
                            content={
                                "error": True,
                                "error_info": "No prompt text was provided"
                            }
                            )
    # generation arguments
    generate_kwargs: dict = body["generate_kwargs"] if isinstance(body["generate_kwargs"], Dict) else None

    with torch.no_grad():
        start = perf_counter()
        generations: List = pipe(batch, **generate_kwargs)
        end = perf_counter()
        elapsed_time = f"{end - start:.5f}s"


    return JSONResponse(status_code=200,
                        content={
                            "generations": generations,
                            "elapsed_time": elapsed_time
                        }
                        )      

@app.get("/status/")
def status():
    return JSONResponse(status_code=200,content={"message" : "API is running"})         

    

if __name__ == '__main__':

    uvicorn.run(
    app="server:app",
    host="0.0.0.0",
    port=5555,
    log_level="info"
    )
