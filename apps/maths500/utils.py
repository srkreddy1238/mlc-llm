import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import time
import pandas as pd


def extract_boxed_solution(text: str) -> str | None:
    """
    Extracts the content of the last `\boxed{}` in a given LaTeX-style text.

    Args:
        text (str): The input string containing LaTeX-style content.

    Returns:
        Optional[str]: The extracted content inside the last `\boxed{}` if found 
        and properly matched, otherwise `None`.

    Example:
        >>> extract_boxed_solution("The result is \\boxed{42}.")
        '42'
        >>> extract_boxed_solution("Unmatched \\boxed{42")
        None
    """
    try:
        start_index = text.rindex("\\boxed{")
        content_start = start_index + 7
        bracket_count = 1
        current_pos = content_start

        while bracket_count > 0 and current_pos < len(text):
            if text[current_pos] == "{":
                bracket_count += 1
            elif text[current_pos] == "}":
                bracket_count -= 1
            current_pos += 1

        if bracket_count == 0:
            content = text[content_start : current_pos - 1].strip()
            return content
        else:
            print("Error: Unmatched brackets in the text")
            return None

    except ValueError:
        print("No boxed solution found in the text")
        return None
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None


def extract_boxed_solution_modified(text: str) -> str | None:
    """
    Extracts the content of any complete `\boxed{}` in a given LaTeX-style text.

    Args:
        text (str): The input string containing LaTeX-style content.

    Returns:
        Optional[str]: The extracted content inside the first complete `\boxed{}` if found 
        and properly matched, otherwise `None`.

    Example:
        >>> extract_boxed_solution("The result is \\boxed{42}.")
        '42'
        >>> extract_boxed_solution("Unmatched \\boxed{42")
        None
    """
    try:
        start_index = 0
        while True:
            start_index = text.find("\\boxed{", start_index)
            if start_index == -1:
                print("No complete boxed solution found in the text")
                return None

            content_start = start_index + 7
            bracket_count = 1
            current_pos = content_start

            while bracket_count > 0 and current_pos < len(text):
                if text[current_pos] == "{":
                    bracket_count += 1
                elif text[current_pos] == "}":
                    bracket_count -= 1
                current_pos += 1

            if bracket_count == 0:
                content = text[content_start : current_pos - 1].strip()
                return content
            else:
                start_index += (
                    7
                )  # Move past the current incomplete \boxed{ to search for the next one

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None


def extract_last_complete_boxed_solution(text: str) -> str | None:
    """
    Extracts the content of the last complete `\boxed{}` in a given LaTeX-style text.

    Args:
        text (str): The input string containing LaTeX-style content.

    Returns:
        Optional[str]: The extracted content inside the last complete `\boxed{}` if found 
        and properly matched, otherwise `None`.

    Example:
        >>> extract_last_complete_boxed_solution("The result is \\boxed{42}.")
        '42'
        >>> extract_last_complete_boxed_solution("Unmatched \\boxed{42")
        None
    """
    try:
        last_complete_content = None
        start_index = 0

        while start_index < len(text):
            start_index = text.find("\\boxed{", start_index)
            if start_index == -1:
                break

            content_start = start_index + 7
            bracket_count = 1
            current_pos = content_start

            while bracket_count > 0 and current_pos < len(text):
                if text[current_pos] == "{":
                    bracket_count += 1
                elif text[current_pos] == "}":
                    bracket_count -= 1
                current_pos += 1

            if bracket_count == 0:
                last_complete_content = text[content_start : current_pos - 1].strip()
                start_index = current_pos
            else:
                start_index += (
                    7
                )  # move past the current incomplete \boxed{ to search for the next one

        if last_complete_content is not None:
            print(last_complete_content)
            return last_complete_content
        else:
            print("No complete boxed solution found in the text")
            return None

    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None


def find_total_tokens(kpi: str) -> tuple:

    start_index = kpi.find("Prompt ", 0)
    if start_index == -1:
        return (0, 0, 0)
    end_index = kpi.find(" toks")
    prompt_tokens = int(kpi[start_index + len("Prompt ") : end_index])

    start_index_line2 = kpi.find("Generate ")
    end_index_line2 = kpi.find(" toks", start_index_line2)
    output_tokens = int(kpi[start_index_line2 + len("Generate ") : end_index_line2])

    total_tokens = prompt_tokens + output_tokens

    start_index_3 = kpi.find("(msec) ", 0)
    end_index_3 = kpi.find(" (t/s)", start_index_3)
    ttft = float(kpi[start_index_3 + len("(msec) ") : end_index_3])

    start_index_4 = kpi.find("(msec) ", start_index_line2)
    end_index_4 = kpi.find(" (t/s)", start_index_4)
    token_rate = float(kpi[start_index_4 + len("(msec) ") : end_index_4])

    return (total_tokens, ttft, token_rate)
