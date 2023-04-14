from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import GenerationConfig

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT


@dataclass
class InferenceArguments:
  model_max_length: int = field(
    default=512,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  load_in_8bit: bool = field(
    default=False,
    metadata={"help": "Load the model in 8-bit mode."},
  )
  inference_dtype: torch.dtype = field(
    default=torch.float32,
    metadata={"help": "The dtype to use for inference."},
  )


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()

  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=inference_args.load_in_8bit,
    torch_dtype=inference_args.inference_dtype,
    device_map="auto",
  )
  model.cuda()
  model.eval()

  generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )
  tokenizer.add_special_tokens(
    {
      "eos_token": DEFAULT_EOS_TOKEN,
      "bos_token": DEFAULT_BOS_TOKEN,
      "unk_token": DEFAULT_UNK_TOKEN,
    }
  )

  ctx = ""
  for instruction in [
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
  ]:
    print("Instruction:", instruction)
    inputs = tokenizer(generate_prompt(instruction, None), return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                             generation_config=generation_config,
                             max_new_tokens=inference_args.model_max_length,
                             return_dict_in_generate=True,
                             output_scores=True)
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    ctx += f"Instruction: {instruction}\n" + f"Response: {generated_tokens[0]}\n"
    print("Response:", tokenizer.decode(generated_tokens[0]))
    print()


if __name__ == "__main__":
  inference()
