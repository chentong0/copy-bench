import json
from tqdm import tqdm
import re
import os

from collections import Counter
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def main(args):
    input_path = args.input
    output_path = args.output
    if os.path.exists(output_path):
        return
    with open(input_path, "r") as f:
        results_list = json.load(f)

    # nlp = spacy.load("en_core_web_sm")
    # text_sim_model = unisim.TextSim()
    # rouge_model = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for inst in tqdm(results_list):
        label_text, output_text = inst["reference"], inst["output"]
        # extract answer from output: [Output_Start] ... [Output_End]
        try:
            pattern = re.compile(r"\[Output_Start\](.*)\[Output_End\]")
            match = pattern.search(output_text)
            if match:
                output_text = match.group(1)
            else:
                output_text = output_text
            
            em = float(exact_match_score(output_text, label_text))
            f1 = f1_score(output_text, label_text)
        except Exception as e:
            em = 0.0
            f1 = 0.0
            print(f'[WARNING] Error in calculating scores: {e}')

        # inst["score_rouge_1"] = rouge_1
        # inst["score_rouge_l"] = rouge_l
        # inst["score_retsim"] = retsim
        # inst["score_lcs"] = lcs
        inst["score_em"] = em
        inst["score_f1"] = f1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2)


if __name__ == "__main__":
    import argparse
    # input_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # parser.add_argument("--words", type=int, default=50)
    args = parser.parse_args()
    main(args)
