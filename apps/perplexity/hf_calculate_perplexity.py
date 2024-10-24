"""
This script is intended for the comparison purpose,
Use this script to calculate the perplexity of models
using pytorch solution on NVIDIA GPUs, we load the model
using 4-Bit quantization from bitsandbytes library that 
is supported in pytorch and can be given as an argument
load_in_4bit = True. 
For the same prompt calculate the score for Qualcomm silicon 
with Machine Learning Compilation solution by TVM.
"""

import torch
import argparse

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, LlamaTokenizer, AutoModelForCausalLM


def calculate_perplexity(logits, targets):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    targets = targets.view(-1)
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    shift_logits = shift_logits.to(shift_logits.device)
    shift_labels = shift_labels.to(shift_labels.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def load_inputs_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        num_inputs = len(lines)
        vectors = torch.zeros((num_inputs, 256), dtype=torch.int64)
        for i, line in enumerate(lines):
            vector = torch.tensor([float(x) for x in line.split()])
            vectors[i, :] = vector
    return vectors

def torch_infer_llm(args):
    inputs_custom = load_inputs_from_file(args.input_path)
    model_name = args.model_path

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        max_memory=max_memory,
        do_sample=True,
        torch_dtype="auto",
    )

    nlls = []
    nlls_tvm = []
    prev_end_loc = 0
    i = 1
    for inputs in inputs_custom:
        input_ids = inputs.unsqueeze(0)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            # neg_log_likelihood = outputs.loss
            neg_log_likelihood_custom_hf = calculate_perplexity(outputs[1], target_ids)
            print("Sample number: ", i, " ", "Loss: ", neg_log_likelihood_custom_hf)
            i += 1
        nlls.append(neg_log_likelihood_custom_hf)
    return nlls


def main():
    parser = argparse.ArgumentParser(description ="Calculate perplexity for HF model using torch")
    required = parser.add_argument_group("required arguments")
    parser.add_argument("--input-path", type=str, help="input file path", required=True)
    parser.add_argument("--model-path", type=str, help="path of hugging-face model", required=True)
    args = parser.parse_args()
    nlls = torch_infer_llm(args)
    ppl_hf = torch.exp(torch.stack(nlls).mean())
    print(ppl_hf)

if __name__ == "__main__":
    main()

"""python3 hf_calculate_perplexity.py --input-path INPUT_PATH --model-path HF_MODEL_PATH"""