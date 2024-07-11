# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging
logging.basicConfig(level=logging.INFO)

import json
from tqdm import tqdm
import torch

class AttributionModel:
    def __init__(self, model_name, device="cuda"):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.hf_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.hf_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16,
        )
        self.device = device
        self.max_length = 512
        if model_name == "osunlp/attrscore-flan-t5-xl":
            self.get_prompt = lambda premise, hypothesis: f"As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {hypothesis} \n Reference: {premise}"
            self.get_class = lambda output: output == "Attributable"
        elif model_name == "google/flan-t5-xl":
            self.get_prompt = lambda premise, hypothesis: f"### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'attributable' or 'not attributable'.\n### Input:\nClaim: {hypothesis}\n\nReference: {premise}\n\n### Output:"
            self.get_class = lambda output: output == "attributable"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.spacy_model = None

    @torch.inference_mode()
    def infer_batch(self, data, batch_size=8):
        # assert isinstance(self.hf_model, T5ForConditionalGeneration), "Only T5 models are supported"
        prompt_bid_list = []
        prompt_list = []
        premise_to_chunks = {}
        for bid, inst in enumerate(data):
            premise, hypothesis = inst["premise"], inst["hypothesis"]
            if premise not in premise_to_chunks:
                premise_to_chunks[premise] = self.to_chunks(premise, window=128, stride=128)
            for chunk in premise_to_chunks[premise]:
                prompt = self.get_prompt(chunk, hypothesis)
                prompt_list.append(prompt)
                prompt_bid_list.append(bid)
        output_list = [False] * len(data)
        for i in tqdm(range(0, len(prompt_list), batch_size)):
            batch_prompt_list = prompt_list[i:i + batch_size]
            inputs = self.hf_tokenizer(
                batch_prompt_list, return_tensors="pt", 
                padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            # print(inputs["input_ids"].shape, inputs["attention_mask"].shape)
            outputs = self.hf_model.generate(**inputs, max_new_tokens=10)
            for j, output in enumerate(outputs):
                output_text = self.hf_tokenizer.decode(output, skip_special_tokens=True)
                bid = prompt_bid_list[i + j]
                output_list[bid] = output_list[bid] or self.get_class(output_text)
                # print(output_text)
                # print(output)
        if sum(output_list) > 0:
            print(output_list)
        return output_list

    def to_chunks(self, text, window=128, stride=64):
        import spacy
        # download the English model: 
        if self.spacy_model is None:
            # Load the English language model
            self.spacy_model = spacy.load("en_core_web_sm")
            self.spacy_model.max_length = 10_000_000
            self.spacy_model.disable_pipe("ner")
        # remove newlines
        text = text.replace("\n", " ")
        # Process the text
        doc = self.spacy_model(text)
        # Extract words from the text with position in the text
        words = [(token.text, token.idx) for token in doc]
        # return words
        chunks = []
        for i in range(0, len(words), stride):
            begin, end = i, min(i + window, len(words))
            begin_idx, end_idx = words[begin][1], words[end - 1][1] + len(words[end - 1][0])
            text_chunk = text[begin_idx:end_idx]
            chunks.append(text_chunk)
        return chunks


def save_data(results_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=2)

def main(args):
    # model_name = "osunlp/attrscore-flan-t5-xl"
    # model_name = "google/flan-t5-xl"
    model_name = args.model_name_or_path
    print(f"[INFO] Loading model {model_name}")
    score_model = AttributionModel(model_name)
    print(f"[INFO] Model {model_name} loaded")

    # all_results_list = []
    # for result_path in result_path_list:
    result_path = args.input
    # output_path = result_path.replace("outputs/outputs/outputs", "outputs/scores/scores")
    output_path = args.output
    # if model_name == "osunlp/attrscore-flan-t5-xl":
    #     # output_path = output_path.replace(".json", ".attrscore.json")
    #     assert output_path.endswith("attrscore.json"), f"Unsupported output path: {output_path}"
    # elif model_name == "google/flan-t5-xl":
    #     # output_path = output_path.replace(".json", ".flan.json")
    #     assert output_path.endswith("flan.json"), f"Unsupported output path: {output_path}"
    # else:
    #     raise ValueError(f"Unsupported model: {model_name}")

    print(f"[INFO] Processing {result_path} -> {output_path}")

    with open(result_path, "r") as f:
        results_list = json.load(f)
    

    for i, inst in enumerate(tqdm(results_list)):
        output_text = inst["output"]
        output_text = output_text if output_text else ""

        referece_text_list = inst["reference_events"]
        inst["reference_events"] = referece_text_list
        events_overlap_list = score_model.infer_batch([{
            "premise": output_text.replace("\n", " "), "hypothesis": event,
        } for event in referece_text_list])
        # events_overlap_rate = sum(events_overlap_list) / len(events_overlap_list)
        inst["event_overlap_list"] = events_overlap_list
        inst["score_event_overlap"] = sum(events_overlap_list)

        if i % 200 == 0:
            save_data(results_list, output_path)
    save_data(results_list, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-xl")
    args = parser.parse_args()
    main(args)

# %%
