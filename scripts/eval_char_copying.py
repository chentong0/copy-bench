# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
logging.basicConfig(level=logging.INFO)

import json
from tqdm import tqdm
import torch

def has_name(text, names):
    import re
    for name in names:
        text_tokens = re.split(r"\W+", text)
        text_tokens = [t.lower() for t in text_tokens if t]
        name_tokens = re.split(r"\W+", name)
        name_tokens = [t.lower() for t in name_tokens if t]
        for i in range(len(text_tokens) - len(name_tokens) + 1):
            if text_tokens[i:i+len(name_tokens)] == name_tokens:
                return True
    return False

def save_data(results_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2)

def main(args):

    result_path = args.input
    output_path = args.output
    # assert output_path.endswith("char.json"), f"Unsupported output path: {output_path}"

    print(f"[INFO] Processing {result_path} -> {output_path}")

    with open(result_path, "r") as f:
        results_list = json.load(f)

    for i, inst in enumerate(tqdm(results_list)):
        output_text = inst["output"]
        output_text = output_text if output_text else ""

        # if title_to_reference is None:
        #     char_list = inst["reference_characters"]
        # else:
        #     char_list = title_to_reference.get(inst["title"], [])
        char_list = inst["reference_characters"]

        input_text = inst["input"]
        # count of characters in the input text
        # input_char_count = sum([has_name(input_text, char) for char in char_list])
        # output_char_count = sum([has_name(output_text, char) for char in char_list])
        # inst["char_overlap"] = output_char_count 
        # char_overlap = sum([max(has_name(output_text, char) - has_name(input_text, char), 0) for char in char_list])
        # inst["char_overlap"] = char_overlap
        # char_overlap_list = [has_name(output_text, char) if not has_name(input_text, char) else None for char in char_list]
        char_overlap_list = []
        for char_names in char_list:
            assert not has_name(input_text, char_names), f"Input text contains {char_names}"
            char_overlap_list.append(has_name(output_text, char_names))

        inst["reference_characters"] = char_list
        inst["char_overlap_list"] = char_overlap_list
        inst["score_char_overlap"] = sum([x for x in char_overlap_list if x is not None])

        if i % 200 == 0:
            save_data(results_list, output_path)
    save_data(results_list, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)

# %%
