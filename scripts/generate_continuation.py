import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from tqdm import tqdm
import random
random.seed(42)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f'[INFO] Save results to {file_path}')

def apply_transform(apply_config, instance):
    instance = instance.copy()
    if "input" in  apply_config:
        assert apply_config["input"] == "capitalize"
        if "input" in instance:
            instance["input"] = instance["input"].upper()
    if "output" in  apply_config:
        assert apply_config["output"] == "capitalize"
        if "output" in instance:
            instance["output"] = instance["output"].upper()
    return instance

def process_conversation(prompt_config, instance, shots=0):
    # sample shots from the instance
    import random, string, re
    demos = random.sample(prompt_config["demos"], shots) if shots > 0 else []
    demo_prompt_template = string.Template(prompt_config["demo_prompt"])
    instruction = prompt_config["instruction"]
    demo_sep = prompt_config["demo_sep"]
    apply_config = prompt_config["apply_config"] if "apply_config" in prompt_config else {}
    demo_prompt_list = [demo_prompt_template.safe_substitute(**apply_transform(apply_config, demo)) for demo in demos]
    prompt = instruction + demo_sep + demo_sep.join(demo_prompt_list)
    if "task_instruction" in prompt_config:
        prompt = prompt + demo_sep + prompt_config["task_instruction"]
    if "task_prompt" in prompt_config:
        task_prompt_template = string.Template(prompt_config["task_prompt"])
        prompt = prompt + demo_sep + task_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    else:
        prompt = prompt + demo_sep + demo_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    # assert no pattern ${...} left in the prompt
    assert re.search(r"\${.*?}", prompt) is None, f"Unresolved pattern in prompt: {prompt}"
    prompt = prompt.strip()
    if prompt_config.get("enable_system", False):
        return [{"role": "system", "content": prompt_config["system"]}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]

# if the model do not support system prompt, combine system prompt with the user instruction
def merge_conversation(conversation):
    # assert len(conversation) <= 2, f"Invalid conversation length: {len(conversation)}"
    # if conversation[0]["role"] == "system":
    #     assert conversation[1]["role"] == "user", f"Invalid conversation roles: {conversation[0]['role']} -> {conversation[1]['role']}"
    #     prompt = conversation[0]["content"] + "\n\n" + conversation[1]["content"]
    #     return [{"role": "user", "content": prompt}]
    # return conversation
    return "\n\n".join([msg["content"] for msg in conversation])


def main(args):
    # input_file = "snippets.json"
    # output_file = "results.json"
    # model_name_or_path = "meta-llama/Llama-2-7b-hf"
    
    with open(args.input_file, "r") as f:
        snippets = json.load(f)
    if args.n_instances is not None:
        # snippets = snippets[:args.n_instances]
        # random sampling
        if args.n_instances < len(snippets):
            snippets = random.sample(snippets, args.n_instances)
        print(f"[INFO] Randomly sampled {args.n_instances} instances from {len(snippets)} instances.")

    # process prompts
    if args.prompt_file is None:
        prompt_config = None
    else:
        with open(args.prompt_file, "r") as f:
            prompt_config = json.load(f)
    assert prompt_config is not None, f"Prompt config is not provided."

    if args.system_prompt:
        prompt_config["enable_system"] = True
    for s in snippets:
        s["conversation"] = process_conversation(prompt_config, s, shots=args.shots)

    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            output_list = json.load(f)
        print(f"[INFO] {len(output_list)} samples already exist in {args.output_file}")
        # remove snippets that are already processed
        existing_ids = [s["id"] for s in output_list]
        snippets = [s for s in snippets if s["id"] not in existing_ids]
    else:
        output_list = []

    if "gpt" in args.model.lower():   # openai api
        from utils.openai_lm import OpenAIModel
        args.batch_size = 10
        model_name = args.model
        # model_name = args.model.replace("gpt-35", "gpt-3.5")
        model = OpenAIModel(model_name, model_mode="chat", api_type="openai")
        print(f"[INFO] Model loaded: {model_name}")
        print(f"[INFO] Test output: {model.generate('how are you?', max_new_tokens=50, temperature=0.0)}")
        with tqdm(total=len(snippets)) as pbar:
            for i in range(0, len(snippets), args.batch_size):
                for s in snippets[i:i+args.batch_size]:
                    try:
                        output_text = model.generate(s["conversation"], max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
                    except Exception as e:
                        print(f"[ERROR] {e}")
                        output_text = ""
                    output_list.append({
                        **s, 
                        "output": output_text,
                    })
                    pbar.update()
                save_json(output_list, args.output_file)
        save_json(output_list, args.output_file)
    else:
        if args.memfree:
            assert isinstance(args.memfree, str), f"Please specify the memfree corpus."
            if args.memfree == "literal":
                rejection_texts = [inst["reference"] for inst in snippets]
            elif args.memfree == "nonliteral":
                with open("../../data/copyright-data/non-literal-rejection-corpus.json", "r") as f:
                    rejection_texts = json.load(f)
            else:
                raise ValueError(f"Invalid memfree corpus: {args.memfree}")
            from vllm_memfree_sampler import vllm_enable_memfree_decoding
            vllm_enable_memfree_decoding(args.model, rejection_texts, rejection_n_gram=args.memfree_n)
        
        from utils.vllm_lm import VllmLM
        model = VllmLM(args.model)
        print(f"[INFO] Model loaded: {args.model}")
        test_output_text_ = model.generate("how are you?", max_new_tokens=50, temperature=0.0)[0]
        print(f"[INFO] Test output: {test_output_text_}")
        with tqdm(total=len(snippets)) as pbar:
            for i in range(0, len(snippets), args.batch_size):
                assert args.format in ["default", "chat", "context"], f"Invalid format: {args.format}"
                prompt_list = []
                if args.format == "chat":
                    tokenizer = model.get_tokenizer()
                    if "tulu" in args.model:
                        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
                    if "vicuna" in args.model:
                        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"

                    prompt_list = [
                        model.get_tokenizer().apply_chat_template(
                            s["conversation"] if not "tulu" in args.model else [{"role": "user", "content": merge_conversation(s["conversation"])}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for s in snippets[i:i+args.batch_size]
                    ]
                else:
                    prompt_list = [merge_conversation(s["conversation"]) for s in snippets[i:i+args.batch_size]]
                output_text_list = model.generate(
                    prompts=prompt_list, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    stop=["\n\n\n"],
                )

                for s, prompt, output_text in zip(snippets[i:i+args.batch_size], prompt_list, output_text_list):
                    output_list.append({
                        **s, 
                        "prompt": prompt,
                        "output": output_text,
                    })
                pbar.update(args.batch_size)
                save_json(output_list, args.output_file)
        save_json(output_list, args.output_file)

        if args.memfree:
            from vllm_memfree_sampler import vllm_disable_memfree_decoding
            vllm_disable_memfree_decoding()
    
    print("[INFO] Exiting...")
    exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_instances", type=int, default=None)

    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--format", choices=["default", "chat", "context"], default="default")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=512)
    # decoding parameters
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    # repetition_penalty
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--memfree", type=str, default=None)
    parser.add_argument("--memfree_n", type=int, default=10)

    parser.add_argument("--system_prompt", action="store_true")

    # parser.add_argument("--copyright", action="store_true")
    # parser.add_argument("--copyright_alpha", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
