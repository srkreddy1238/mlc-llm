import os
import pandas as pd
import subprocess
import re
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=str, help="model artifacts path")
parser.add_argument("--model", type=str, help="model artifacts path")
parser.add_argument("--device", type=str, help="andoid device id")
parser.add_argument("--with-accl", type=bool, default=False)
args = parser.parse_args()

# MLC GenAI supported models
MODELS = [
    "Llama-2-7b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Qwen-7B-Chat",
    "Mistral-7B-Instruct-v0.2",
    "gemma-2b-it",
    "phi-2",
    "Phi-3-mini-4k-instruct",
    "llava-1.5-7b-hf",
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Llama-8B",
    "Phi-3.5-mini-instruct",
]
dist_dir = args.model + "/dist"
mlc_binary = args.bin

Prompt256 = dict()
Prompt256["Llama-2-7b-chat-hf"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath.\\\""
Prompt256["Meta-Llama-3-8B-Instruct"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. A collective gasp went up. And when I emerged in one piece.\\\""
Prompt256["Qwen-7B-Chat"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star.\\\""
Prompt256["Mistral-7B-Instruct-v0.2"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it.\\\""
Prompt256["gemma-2b-it"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. A collective gasp went up. And when I emerged in one piece, I was a star.I was a star. I was a star.\\\""
Prompt256["phi-2"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath.\\\""
Prompt256["Phi-3-mini-4k-instruct"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up.\\\""
Prompt256["llava-1.5-7b-hf"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. I was a star.\\\""
Prompt256["DeepSeek-R1-Distill-Qwen-1.5B"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. A collective gasp went up. And when I emerged in one piece, I was a star.I was a star. I was a star.\\\""
Prompt256["DeepSeek-R1-Distill-Llama-8B"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up. And when I emerged in one piece, I was a star. A collective gasp went up. And when I emerged in one piece, I was a star. I was a star. I was a star.\\\""
Prompt256["Phi-3.5-mini-instruct"] = "\\\"Rewrite story: Come on! It’s starting! Greg, my neighbor, hollered from the sidewalk.  What’s starting? I said. Behind him, groups of kids hurried down the street. We’d moved to the neighborhood just weeks before. I was shy; a bookworm, waiting for school to start. Greg was the only kid I’d met. The magic show! said Greg, exasperated. At Mr. Hale’s house! At the end of the Hales’ dirt driveway, rows of kids were seated on the grass. White-haired and very thin, Mr. Hale wore a black top-hat and tails. In his hand he gripped a wand, producing doves from an urn. He asked for a volunteer to be sawed in half. I raised my hand. No one breathed. Just relax, Mr. Hale whispered. There’s nothing to it. I got into the box and held my breath. A collective gasp went up.\\\""

out_table = dict()
out_table["Model"] = []


os.system("adb -s {device_id} shell \"rm -rf /data/local/tmp/mlc-ci\"".format(device_id=args.device))
os.system("adb -s {device_id} push {mlc_binary} /data/local/tmp/mlc-ci/".format(mlc_binary=mlc_binary, device_id=args.device))

def run_model(prompt, model, prompt_size):
    try:
        exec_cmd = "adb -s {device_id} shell \"cd /data/local/tmp/mlc-ci; LD_LIBRARY_PATH=./lib/ ./bin/mlc_cli_chat --model /data/local/tmp/mlc-ci/models/{model}-q4f16_0-MLC --model-lib /data/local/tmp/mlc-ci/models/{model}-q4f16_0-adreno.so --device opencl --max-tokens 100 --with-prompt {prompt}\"".format(device_id=args.device, model=model, prompt=prompt)
        xx = str(subprocess.Popen(exec_cmd, shell=True, stdout=subprocess.PIPE).stdout.read())
        decode_tok_per_sec = float(re.search(r'decode : \d+.\d+', str(xx)).group().split(" ")[-1])
        prefill_tok_per_sec = float(re.search(r'prefill : \d+.\d+', str(xx)).group().split(" ")[-1])
        prefill_tokens = int(re.search(r'prefill : \d+.\d+ tok/sec \(\d+', str(xx)).group().split("(")[-1])
    except Exception as e:
        print("Error in run model - "+ model + " prompt size - " + str(prompt_size))
        decode_tok_per_sec = -1
        prefill_tokens = -1
        prefill_tok_per_sec = -1
        print(f"An error occurred: {e}")

    if (str(prompt_size) + "-decode token per sec") in out_table.keys():
        out_table[str(prompt_size) + "-decode token per sec"].append(decode_tok_per_sec)
    else:
        out_table[str(prompt_size) + "-decode token per sec"] = [decode_tok_per_sec]
    if (str(prompt_size) + "-prefill tokens") in out_table.keys():
        out_table[str(prompt_size) + "-prefill tokens"].append(prefill_tokens)
    else:
        out_table[str(prompt_size) + "-prefill tokens"] = [prefill_tokens]
    if (str(prompt_size) + "-prefill token per sec") in out_table.keys():
        out_table[str(prompt_size) + "-prefill token per sec"].append(prefill_tok_per_sec)
    else:
        out_table[str(prompt_size) + "-prefill token per sec"] = [prefill_tok_per_sec]
    if (str(prompt_size) + "-prefill TTFT") in out_table.keys():
        out_table[str(prompt_size) + "-prefill TTFT"].append(prefill_tokens/prefill_tok_per_sec)
    else:
        out_table[str(prompt_size) + "-prefill TTFT"] = [prefill_tokens/prefill_tok_per_sec]

    if args.with_accl:
        try:
            exec_cmd = "adb -s {device_id} shell \"cd /data/local/tmp/mlc-ci; LD_LIBRARY_PATH=./lib/ ./bin/mlc_cli_chat --model /data/local/tmp/mlc-ci/models/{model}-q4f16_0-MLC --model-lib /data/local/tmp/mlc-ci/models/{model}-q4f16_0-adreno-accl.so --device opencl --max-tokens 100 --with-prompt {prompt}\"".format(device_id=args.device, model=model, prompt=prompt)
            xx = str(subprocess.Popen(exec_cmd, shell=True, stdout=subprocess.PIPE).stdout.read())
            decode_tok_per_sec = float(re.search(r'decode : \d+.\d+', str(xx)).group().split(" ")[-1])
            prefill_tok_per_sec = float(re.search(r'prefill : \d+.\d+', str(xx)).group().split(" ")[-1])
            prefill_tokens = int(re.search(r'prefill : \d+.\d+ tok/sec \(\d+', str(xx)).group().split("(")[-1])
            print("decode_tokens_per_s - " + str(decode_tok_per_sec))
            print("prefill_tokens_per_s - " + str(prefill_tok_per_sec))
        except Exception as e:
            print("Error in run model - "+ model + " prompt size - " + str(prompt_size))
            decode_tok_per_sec = -1
            prefill_tokens = -1
            prefill_tok_per_sec = -1
            print(f"An error occurred: {e}")
        if (str(prompt_size) + "-with ACCL decode token per sec") in out_table.keys():
            out_table[str(prompt_size) + "-with ACCL decode token per sec"].append(decode_tok_per_sec)
        else:
            out_table[str(prompt_size) + "-with ACCL decode token per sec"] = [decode_tok_per_sec]
        if (str(prompt_size) + "-with ACCL prefill tokens") in out_table.keys():
            out_table[str(prompt_size) + "-with ACCL prefill tokens"].append(prefill_tokens)
        else:
            out_table[str(prompt_size) + "-with ACCL prefill tokens"] = [prefill_tokens]
        if (str(prompt_size) + "-with ACCL prefill token per sec") in out_table.keys():
            out_table[str(prompt_size) + "-with ACCL prefill token per sec"].append(prefill_tok_per_sec)
        else:
            out_table[str(prompt_size) + "-with ACCL prefill token per sec"] = [prefill_tok_per_sec]
        if (str(prompt_size) + "-with ACCL prefill TTFT") in out_table.keys():
            out_table[str(prompt_size) + "-with ACCL prefill TTFT"].append(prefill_tokens/prefill_tok_per_sec)
        else:
            out_table[str(prompt_size) + "-with ACCL prefill TTFT"] = [prefill_tokens/prefill_tok_per_sec]

        print(str(prompt_size) + " Prompt execution done")

for model in MODELS:
    print(model + " execution is in progress ..")
    out_table["Model"].append(model)
    os.system("adb -s {device_id} shell \"rm -rf /data/local/tmp/mlc-ci/models\"".format(device_id=args.device))
    os.system("adb -s {device_id} shell \"mkdir -p /data/local/tmp/mlc-ci/models\"".format(device_id=args.device))
    os.system("adb -s {device_id} push {dist_dir}/{model}-q4f16_0-MLC /data/local/tmp/mlc-ci/models/".format(dist_dir=dist_dir, device_id=args.device, model=model))
    os.system("adb -s {device_id} push {dist_dir}/libs/{model}-q4f16_0-adreno.so /data/local/tmp/mlc-ci/models/".format(dist_dir=dist_dir, device_id=args.device, model=model))
    os.system("adb -s {device_id} push {dist_dir}/libs/{model}-q4f16_0-adreno-accl.so /data/local/tmp/mlc-ci/models/".format(dist_dir=dist_dir, device_id=args.device, model=model))
    time.sleep(60)
    prompt = "\\\"write a story about moon in 100 words.\\\""
    run_model(prompt, model, 100)
    time.sleep(60)
    prompt = Prompt256[model]
    run_model(prompt, model, 256)
    time.sleep(60)

df = pd.DataFrame(data=out_table)
df.to_csv("mlc_llm_perf_"+args.device+".csv")


