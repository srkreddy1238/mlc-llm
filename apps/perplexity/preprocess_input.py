import torch
import argparse
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm

def create_input(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = 256
    stride = 128
    seq_len = encodings.input_ids.size(1)
    device = "cuda"

    nlls = []
    prev_end_loc = 0

    with open(args.destination, "w") as file:
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            data = " ".join(map(str, input_ids.view(-1).tolist()))
            file.write(data + "\n")

def main():
    parser = argparse.ArgumentParser(description ="Create inputs for perplexity calculation")
    required = parser.add_argument_group("required arguments")
    parser.add_argument("--destination", type=str, help="file path for input dump", required=True)
    parser.add_argument("--model_path", type=str, help="path of hugging-face model", required=True)
    args = parser.parse_args()
    create_input(args)

if __name__ == "__main__":
    main()

"""
python3 preprocess_input.py --destination INPUT_FILE_PATH --model_path HF_MODEL_PATH
"""