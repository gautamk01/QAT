import sys
import os

# --- Path Fix ---
project_path = '.'
if project_path not in sys.path:
    sys.path.append(project_path)
# --- End of Path Fix ---

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize.int_linear_fake import QuantLinear
from main_block_ap import evaluate
import utils
import argparse
import types
import gc

# --- Hotfix for peft/accelerate version mismatch ---
mem_mod = sys.modules.get("accelerate.utils.memory") or types.ModuleType("accelerate.utils.memory")
if not hasattr(mem_mod, "clear_device_cache"):
    def clear_device_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    mem_mod.clear_device_cache = clear_device_cache
    sys.modules["accelerate.utils.memory"] = mem_mod
# --- End of Hotfix ---


# --- Configuration ---
model_path = "./quantized_model_w4"
log_dir = "./logs_manual_eval"
eval_tasks_str = "piqa,arc_easy,hellaswag"
# --- End of Configuration ---

# Create a Namespace object to mimic the command-line arguments
args = argparse.Namespace(
    model_path=model_path,
    eval_ppl=True,
    eval_tasks=eval_tasks_str,
    eval_batch_size=16,
    max_memory="70GiB",
    ppl_seqlen=2048
)

# Create the logging directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up a logger
logger = utils.create_logger(log_dir)
logger.info(f"Loading fake quantized model from {args.model_path} for manual evaluation.")

# Load the model and tokenizer
# The key fix is adding trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, legacy=False)

logger.info("Model and tokenizer loaded successfully.")
logger.info(f"Memory footprint after loading: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")

# Run the evaluation function
evaluate(model, tokenizer, args, logger)

logger.info("Evaluation complete.")
