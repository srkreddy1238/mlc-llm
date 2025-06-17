import datasets
import json
import re
import random

# ------------------------------
# STEP 1: Load the MATH500 Dataset
# ------------------------------
dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")  # Check Hugging Face for availability
# ------------------------------
# STEP 2: Function to Format Math Prompts
# ------------------------------
def format_math_problem(entry):
    """Formats a math problem into a structured prompt for MLC-LLM."""
    prompt = f"""You are a mathematical expert. Solve the following problem step by step and provide the final answer in the format: 'The answer is ...'.
Question:
{entry["problem"]}
Provide only the final answer in the format: 'The answer is ...'."""
    return prompt


# ------------------------------
# STEP 3: Dump Formatted Prompts to File
# ------------------------------
output_file = "mlc_math500_prompts.txt"
formatted_text_prompts = []
test_entries = []  # Store dataset entries for answer checking
for entry in dataset["test"]:  # Use test split for evaluation
    prompt = format_math_problem(entry)
    formatted_text_prompts.append(prompt)
    test_entries.append(entry)  # Store for later evaluation
# Save prompts to a text file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(formatted_text_prompts))
print(f"✅ All formatted math prompts saved to {output_file}.")
# ------------------------------
# STEP 4: Read Prompts from File
# ------------------------------
with open(output_file, "r", encoding="utf-8") as f:
    all_prompts = f.read().strip()
# Split prompts using separator ---
prompts = all_prompts.split("\n\n---\n\n")
