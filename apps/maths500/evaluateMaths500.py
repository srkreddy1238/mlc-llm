import re
import sympy as sp
from apps.maths500.grader import grade_answer


def extract_math_answer(response):
    """
    Extracts the final numerical or symbolic answer from model output.
    1️⃣ First, checks for \boxed{...} (preferred format).
    2️⃣ If no \boxed{...}, falls back to "The answer is ...".
    3️⃣ Returns None if no valid answer is found.
    """
    response_content = response.split("=== RESPONSE END ===")[
        0
    ].strip()  # Keep only relevant section
    # 1️⃣ Check for boxed answers (preferred method)
    boxed_matches = re.findall(r"\\boxed{(.+?)}", response_content)
    if boxed_matches:
        return boxed_matches[-1].strip()  # Take the last occurrence of \boxed{...}
    # 2️⃣ If no \boxed{}, fallback to extracting "The answer is ..."
    answer_matches = re.findall(r"The answer is (.+)", response_content)
    if answer_matches:
        return answer_matches[-1].strip()  # Take the last occurrence of "The answer is ..."
    return None  # If neither format is found, return None


def normalize_math_expression(expression):

    """
   Converts LaTeX-style expressions and standardizes math notation for comparison.
   Uses SymPy to parse and compare expressions correctly.
   Removes units, special characters, and trailing punctuation if necessary.
   Handles tuple formatting consistency.
   """
    try:
        # Remove special characters like $, %, and trailing non-numeric text
        expression = re.sub(r"[^\d./,()-]+$", "", expression).strip()
        # Remove unnecessary spaces inside tuples (e.g., "(-2, 1)" vs. "(-2,1)")
        expression = re.sub(r"\(\s*", "(", expression)
        expression = re.sub(r"\s*\)", ")", expression)
        expression = re.sub(r"\s*,\s*", ",", expression)
        # Remove trailing punctuation like . , $
        expression = expression.rstrip(".$,")
        return str(sp.sympify(expression))  # Convert LaTeX-style math to standard form
    except:
        return expression  # Return as-is if conversion fails


def evalMaths500():
    # ------------------------------
    # STEP 1: Read Input Files
    # ------------------------------
    response_file = "maths500Output.txt"  # Model-generated responses
    answer_file = "correct_answers.txt"  # Ground truth answers
    evaluation_file = "evaluation_results.txt"
    with open(response_file, "r", encoding="utf-8") as f:
        responses = f.read().split("=== RESPONSE START ===")[
            1:
        ]  # Ignore content before first marker
    with open(answer_file, "r", encoding="utf-8") as f:
        correct_answers = [line.strip() for line in f.readlines()]

    # ------------------------------
    # STEP 2: Extract Model Answers & Compare
    # ------------------------------
    correct_count = 0
    total_questions = len(correct_answers)
    results = []
    for i in range(total_questions):
        if i >= len(responses):  # If output.txt has fewer responses, break
            break
        model_response = responses[i].strip()  # Get the model's full response
        predicted_answer = extract_math_answer(model_response)  # Extract the final answer
        correct_answer = correct_answers[i]  # Ground truth answer
        is_correct = False  # Default to incorrect
        norm_predicted = None
        norm_correct = normalize_math_expression(correct_answer)  # Normalize ground truth answer
        if predicted_answer is not None:
            norm_predicted = normalize_math_expression(predicted_answer)
            # Match while ignoring formatting, special characters, units, punctuation, tuple spacing, and minor variations
            is_correct = norm_predicted == norm_correct
            if is_correct:
                correct_count += 1
        results.append(
            {
                "question_id": i + 1,
                "full_model_response": model_response,
                "predicted_answer": predicted_answer,
                "normalized_predicted": norm_predicted if predicted_answer is not None else "N/A",
                "correct_answer": correct_answer,
                "normalized_correct": norm_correct,
                "correct": is_correct,
            }
        )
    # -----------------------------
    # ------------------------------
    # STEP 3: Save Evaluation Results
    # ------------------------------
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    # ------------------------------
    # STEP 3: Save Evaluation Results
    # ------------------------------
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    with open(evaluation_file, "w", encoding="utf-8") as f:
        f.write(
            f"✅ Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions} correct)\n\n"
        )
        for res in results:
            f.write(f"Question ID: {res['question_id']}\n")
            f.write(f"Full Model Response: {res['full_model_response']}\n")
            f.write(f"Extracted Model Answer: {res['predicted_answer']}\n")
            f.write(f"Normalized Predicted Answer: {res['normalized_predicted']}\n")
            f.write(f"Correct Answer: {res['correct_answer']}\n")
            f.write(f"Normalized Correct Answer: {res['normalized_correct']}\n")
            f.write(f"Correct: {'Yes' if res['correct'] else 'No'}\n")
            f.write("\n---\n\n")
        # print(f"\n✅ Accuracy evaluation saved to {evaluation_file}.")
        # print(f"✅ Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions} correct)")

    final_res = []
    false_res = []
    for res in results:
        if res["correct"] == False and res["predicted_answer"] != None:
            final_res.append(
                {
                    "question_id": res["question_id"],
                    "predicted_answer": res["predicted_answer"],
                    "correct_answer": res["correct_answer"],
                }
            )

        # elif res["predicted_answer"] == None:
        #   final_res.append({
        #       "question_id": res['question_id'],
        #       "predicted_answer": "Error (due to insufficient context)",
        #       "correct_answer": res["correct_answer"],
        #   })
        elif res["correct"] == True and res["predicted_answer"] != None:
            final_res.append(
                {
                    "question_id": res["question_id"],
                    "predicted_answer": res["predicted_answer"],
                    "correct_answer": res["correct_answer"],
                }
            )

    import pandas as pd

    df = pd.DataFrame(final_res)
    df.to_excel("output_new.xlsx", index=False)

    df = pd.read_excel("output_new.xlsx")
    count = 0
    for index, row in df.iterrows():
        if index == 0:
            continue
        val1 = row[1]
        val2 = row[2]
        if grade_answer(str(val1), str(val2)):
            count += 1
    accuracy_wo_CLmiss = count / (len(df) - 1)
    effective_accuracy = count / 500
    print("Accuracy excluding CL exceeded samples:- ", accuracy_wo_CLmiss * 100, " %")
    print("Effective Accuracy:- ", effective_accuracy * 100, " %")
    return 0
