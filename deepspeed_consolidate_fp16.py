import os
import sys

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import torch


assert len(sys.argv) == 2
model_path = sys.argv[1]
print(f"model_path = {model_path}")

state_dict = get_fp32_state_dict_from_zero_checkpoint(model_path)
for key in state_dict:
    state_dict[key] = state_dict[key].half()

output_path = os.path.join(model_path, "pytorch_model.bin")
torch.save(state_dict, output_path)
print(f"{output_path} is saved successfully.")
