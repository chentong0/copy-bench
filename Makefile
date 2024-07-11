export PYTHONPATH=./src:$$PYTHONPATH

quick_start:
	$(eval HF_MODEL=meta-llama/Meta-Llama-3-8B)
	$(eval MODEL_TAG=llama3-8b)
	$(eval N=10)

	### Evaluating literal copying
	python scripts/generate.py \
		--input_file data/data.literal.json \
		--prompt_file prompts/prompts.literal.format1.json \
		--output_file outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--n_instances $(N) \

	python scripts/eval_literal_copying.py \
		--input outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-literal-copying.literal.prompt1.$(MODEL_TAG).greedy.json

	python scripts/eval_quality.py \
		--input outputs/outputs.literal.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-quality.literal.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate non-literal copying
	python scripts/generate.py \
		--input_file data/data.nonliteral.json \
		--prompt_file prompts/prompts.nonliteral.format1.json \
		--output_file outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--max_new_tokens 1024 \
		--n_instances $(N) \

	python scripts/eval_char_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-char-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	python scripts/eval_event_copying.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-event-copying.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	python scripts/eval_quality.py \
		--input outputs/outputs.nonliteral.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-quality.nonliteral.prompt1.$(MODEL_TAG).greedy.json

	#### Evaluate Fact Recall
	python scripts/generate.py \
		--input_file data/data.qa.json \
		--prompt_file prompts/prompts.qa.format1.json \
		--output_file outputs/outputs.qa.prompt1.$(MODEL_TAG).greedy.json \
		--model $(HF_MODEL) \
		--n_instances $(N) \

	python scripts/eval_qa.py \
		--input outputs/outputs.qa.prompt1.$(MODEL_TAG).greedy.json \
		--output scores/scores-qa.qa.prompt1.$(MODEL_TAG).greedy.json

	python scripts/summary.py --root ./scores/**.json
