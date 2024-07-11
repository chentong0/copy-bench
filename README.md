# CopyBench

This repository includes the original implementation of the paper [CopyBench: Measuring Literal and Non-Literal Reproduction of Copyright-Protected Text in Language Model Generation](https://www.arxiv.org/abs/2407.07087) by Tong Chen, Akari Asai*, Niloofar Mireshghallah*, Sewon Min, James Grimmelmann, Yejin Choi, Hannaneh Hajishirzi, Luke Zettlemoyer, Pang Wei Koh. (*Equal contribution)

## Installation
Install dependent Python libraries by running the command below.
```
pip install -r requirements.txt
```

## Quick Start

Evaluate the Llama3-8B model on the following aspects:

- Literal copying
- Non-literal copying
- Fact recall
- Fluency


### Setup Environment Variables

- `PYTHONPATH`: This variable is configured to include the ./src directory. It ensures that Python can locate and import utility functions and modules stored in the src directory, which are crucial for the project's functionality.
- `HF_MODEL` and `MODEL_TAG`: These variables replace placeholders in commands throughout this section.

```bash
export PYTHONPATH=./src:$PYTHONPATH
export HF_MODEL=meta-llama/Meta-Llama-3-8B
export MODEL_TAG=llama3-8b
```

### Literal Copying

```bash
## Generate outputs to evaluate literal copying
python scripts/generate.py \
      --input_file data/data.literal.json \
      --prompt_file prompts/prompts.literal.format1.json \
      --output_file outputs/outputs.literal.prompt1.${MODEL_TAG}.greedy.json \
      --model ${HF_MODEL} \

## Evaluate literal copying in the generated outputs
python scripts/eval_literal_copying.py \
      --input outputs/outputs.literal.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-literal-copying.literal.prompt1.${MODEL_TAG}.greedy.json
```

### Non-literal Copying

```bash
## Generate outputs to evaluate non-literal copying
python scripts/generate.py \
      --input_file data/data.nonliteral.json \
      --prompt_file prompts/prompts.nonliteral.format1.json \
      --output_file outputs/outputs.nonliteral.prompt1.${MODEL_TAG}.greedy.json \
      --model ${HF_MODEL} \
      --max_new_tokens 1024 \

## Evaluate character overlap in the generated outputs
python scripts/eval_char_copying.py \
      --input outputs/outputs.nonliteral.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-char-copying.nonliteral.prompt1.${MODEL_TAG}.greedy.json

## Evaluate event overlap in the generated outputs
python scripts/eval_event_copying.py \
      --input outputs/outputs.nonliteral.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-event-copying.nonliteral.prompt1.${MODEL_TAG}.greedy.json
```

### Fact Recall

```bash
## Generate outputs to evaluate fact recall
python scripts/generate.py \
      --input_file data/data.qa.json \
      --prompt_file prompts/prompts.qa.format1.json \
      --output_file outputs/outputs.qa.prompt1.${MODEL_TAG}.greedy.json \
      --model ${HF_MODEL} \

## Evaluate the accuracy of the generated responses
python scripts/eval_qa.py \
      --input outputs/outputs.qa.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-qa.qa.prompt1.${MODEL_TAG}.greedy.json
```

### Fluency

```bash
## Evaluate the fluency of the outputs in literal copying evaluation
python scripts/eval_fluency.py \
      --input outputs/outputs.literal.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-fluency.literal.prompt1.${MODEL_TAG}.greedy.json

## Evaluate the fluency of the outputs in non-literal copying evaluation
python scripts/eval_fluency.py \
      --input outputs/outputs.nonliteral.prompt1.${MODEL_TAG}.greedy.json \
      --output scores/scores-fluency.nonliteral.prompt1.${MODEL_TAG}.greedy.json
```

### Summarize the Results for Each Model 

```bash
python scripts/summary.py --root ./scores/**.json
```

## Generate Response with Copyright Risk Mitigation Methods
Comming soon.

## Citation
If you find our code, data, or the paper useful, please cite the paper:
```
@misc{chen2024copybenchmeasuringliteralnonliteral,
      title={CopyBench: Measuring Literal and Non-Literal Reproduction of Copyright-Protected Text in Language Model Generation}, 
      author={Tong Chen and Akari Asai and Niloofar Mireshghallah and Sewon Min and James Grimmelmann and Yejin Choi and Hannaneh Hajishirzi and Luke Zettlemoyer and Pang Wei Koh},
      year={2024},
      eprint={2407.07087},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.07087}, 
}
```
