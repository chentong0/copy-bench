# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
logging.basicConfig(level=logging.INFO)

import json
from tqdm import tqdm
import torch

from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import torch

def save_data(results_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2)

def main(args):
    result_path = args.input
    output_path = args.output

    print(f"[INFO] Processing {result_path} -> {output_path}")

    with open(result_path, "r") as f:
        results_list = json.load(f)
    
    judge = PrometheusEval(
        model_id="prometheus-eval/prometheus-7b-v2.0", 
        absolute_grade_template=ABSOLUTE_PROMPT, 
        num_gpus=torch.cuda.device_count(),
        # dtype="auto" if torch.cuda.is_bf16_supported() else "half",
        dtype="half",
    )

    # instruction = "Compose an original story about 500 words in length"
    # rubric_data = {
        # "criteria": "Evaluate the overall quality of a generated story based on coherence, logic, and engagement.",
        # "score1_description": "The story is highly incoherent, with numerous logical inconsistencies, repetitive plots, and chaotic scenes. It fails to engage the reader due to its confusing narrative.",
        # "score2_description": "The story has noticeable coherence issues, including some logical inconsistencies and repetitive elements. It partially engages the reader but still suffers from a confusing narrative.",
        # "score3_description": "The story is generally coherent but contains minor logical inconsistencies or repetitive elements. It engages the reader to some extent, though certain parts may be confusing or lack depth.",
        # "score4_description": "The story is mostly coherent with occasional minor issues. It has a clear narrative and logical flow, engaging the reader effectively despite some minor shortcomings.",
        # "score5_description": "The story is highly coherent, with a clear, logical flow and no significant issues. It is engaging and well-constructed, maintaining the reader's interest throughout."
    # }
    instruction = ""
    rubric_data = {
        "criteria": "Evaluate whether the given text is well-written and grammatical.",
        "score1_description": "The response is completely unclear, making comprehension difficult.",
        "score2_description": "The response has significant areas of ambiguity or disorganization, critically affecting reader comprehension.",
        "score3_description": "The response contains some unclear components, or its organization could be improved.",
        "score4_description": "The response is generally understandable but could be further optimized for readability.",
        "score5_description": "The response is clear and well-organized, enabling the reader to effortlessly follow the content."
    }
#     Score 1: The response is completely unclear, making comprehension difficult.
# Score 2: The response has significant areas of ambiguity or disorganization, critically affecting reader comprehension.
# Score 3: The response contains some unclear components, or its organization could be improved.
# Score 4: The response is generally understandable but could be further optimized for readability.
# Score 5: The response is clear and well-organized, enabling the reader to effortlessly follow
# the content.
    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    output_text_list = [inst["output"] for inst in results_list]
    instruction_list = [instruction] * len(output_text_list)


    # Absolute Grading: Outputs score of 1 to 5
    feedbacks, scores = judge.absolute_grade(
        instructions=instruction_list,
        responses=output_text_list,
        rubric=score_rubric,
        params={}
    )
    # print("Feedback:", feedbacks)
    # print("Score:", scores)
    for i, inst in enumerate(tqdm(results_list)):
        inst["score_fluency"] = scores[i]
        inst["fluency_feedback"] = feedbacks[i]
    save_data(results_list, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)

# %%
