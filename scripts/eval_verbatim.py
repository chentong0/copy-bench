# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# import unisim
from rouge_score import rouge_scorer

# # first_k_words = 50

# # text_sim_model = unisim.TextSim()
# rouge_model = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# all_results_list = []
# for result_path in result_path_list:
#     # output_path = result_path.replace("results/outputs/outputs", "results/scores/scores")
#     output_path = result_path.replace("results/outputs/outputs", "results/scores/scores-lcs")
#     # first_k_words = 100

#     if not os.path.exists(output_path):
#         with open(result_path, "r") as f:
#             results_list = json.load(f)

#         for inst in tqdm(results_list):
#             label_text, output_text = inst["label"], inst["output"]

#             output_text = " ".join(output_text.split()[:first_k_words])
#             label_text = " ".join(label_text.split()[:first_k_words])
            
#             output_text = output_text.replace("_", " ")

#             try:
#                 rouge_dict = rouge_model.score(output_text, label_text)
#                 rouge_1 = rouge_dict["rouge1"].fmeasure
#                 rouge_l = rouge_dict["rougeL"].fmeasure
#                 lcs = rouge_dict["rougeL"].lcs

#                 # retsim = text_sim_model.similarity(output_text, label_text)
#                 retsim = 0.0
#             except Exception as e:
#                 rouge_1 = 0.0
#                 rouge_l = 0.0
#                 retsim = 0.0
#                 lcs = 0.0
#                 # raise e

#             inst["score_rouge_1"] = rouge_1
#             inst["score_rouge_l"] = rouge_l
#             inst["score_retsim"] = retsim
#             inst["score_lcs"] = lcs

#         with open(output_path, "w") as f:
#             json.dump(results_list, f, indent=2)

#     with open(output_path, "r") as f:
#         results_list = json.load(f)
#     # results_list = results_list[:1024]
#     # format: results.(domain).(model).(decoding).json
#     # pattern = re.compile(r"scores\.(.*)\.(.*)\.(.*)\.json")
#     pattern = re.compile(r"scores-lcs\.(.*)\.(.*)\.(.*)\.json")
#     match = pattern.match(os.path.basename(output_path))
#     domain, model, decoding = match.groups()
#     all_results_list.extend([{
#         "domain": domain, "model": model, "decoding": decoding,
#         **inst
#     } for inst in results_list
#     # if inst["id"].split(".")[-1] == "00"
#     ])

# results_df = pd.DataFrame(all_results_list)
# print(results_df.shape)
# results_df.head(5)


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