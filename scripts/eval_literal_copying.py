# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# import unisim
from rouge_score import rouge_scorer

def main(args):
    input_path = args.input
    output_path = args.output
    if os.path.exists(output_path):
        return
    with open(input_path, "r") as f:
        results_list = json.load(f)

    # nlp = spacy.load("en_core_web_sm")
    # text_sim_model = unisim.TextSim()
    rouge_model = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for inst in tqdm(results_list):
        label_text, output_text = inst["reference"], inst["output"]
        try:
            # truncate label_text to first k words
            label_tokens = rouge_model._tokenizer.tokenize(label_text)[:args.words]
            output_tokens = rouge_model._tokenizer.tokenize(output_text)[:args.words]
            # truncate label_text to first k words
            rouge_l_dict = rouge_scorer._score_lcs(label_tokens, output_tokens)
            lcs_table = rouge_scorer._lcs_table(label_tokens, output_tokens)

            rouge_l = rouge_l_dict.fmeasure
            lcs = lcs_table[-1][-1]

            rouge_dict = rouge_model.score(output_text, label_text)
            rouge_1 = rouge_dict["rouge1"].fmeasure

            # retsim = text_sim_model.similarity(output_text, label_text)
            # retsim = 0.0
        except Exception as e:
            rouge_1 = 0.0
            rouge_l = 0.0
            # retsim = 0.0
            lcs = 0.0
            # raise e

        inst["score_rouge_1"] = rouge_1
        inst["score_rouge_l"] = rouge_l
        # inst["score_retsim"] = retsim
        inst["score_lcs"] = lcs

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2)


if __name__ == "__main__":
    import argparse
    # input_path, output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--words", type=int, default=50)
    args = parser.parse_args()
    main(args)
