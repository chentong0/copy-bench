class VllmLM:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None):
        from transformers import AutoConfig
        import vllm
        import torch
        use_slow_tokenizer = False
        if "olmo" in model_name_or_path.lower():
            import hf_olmo
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model_vllm = vllm.LLM(
            model=model_name_or_path,
            tokenizer=tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path,
            tokenizer_mode="slow" if use_slow_tokenizer else "auto",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=self.get_gpu_memory_utilization(),
            # dtype="auto" if torch.cuda.is_bf16_supported() else "half",
            dtype="half",
            trust_remote_code=True,
            max_model_len=min(8192, config.max_position_embeddings),
        )
        self.tokenizer = self.model_vllm.get_tokenizer()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_tokenizer(self):
        return self.model_vllm.get_tokenizer()

    def generate(self, prompts=None, prompt_token_ids=None, max_new_tokens=1024, temperature=0.0, **sampleing_kwargs):
        import vllm
        assert "max_tokens" not in sampleing_kwargs, "use max_new_tokens instead of max_tokens."
        assert prompts is not None or prompt_token_ids is not None, "Either prompts or prompt_token_ids must be provided."
        assert not (prompts is not None and prompt_token_ids is not None), "Either prompts or prompt_token_ids must be provided, but not both."
        if isinstance(prompts, str):
            prompts = [prompts]
        sampling_params = vllm.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            **sampleing_kwargs
        )
        # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
        # Print the outputs.
        output_text_list = []
        outputs = self.model_vllm.generate(prompts=prompts, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
        for i, output in enumerate(outputs):
            # prompt = output.prompt
            if prompts is not None:
                assert output.prompt == prompts[i]
            elif prompt_token_ids is not None:
                assert len(output.prompt_token_ids) == len(prompt_token_ids[i])
                assert all(a == b for a, b in zip(output.prompt_token_ids, prompt_token_ids[i]))
            else:
                raise ValueError("Either prompts or prompt_token_ids must be provided.")
            generated_text = output.outputs[0].text
            output_text_list.append(generated_text)
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return output_text_list

    @staticmethod
    def get_gpu_memory_utilization():
        """
        Get the current GPU memory utilization.
        """
        import torch
        gpu_max_memory = torch.cuda.mem_get_info(0)[1] / 1024**3
        return min(0.9, (gpu_max_memory - 3) / gpu_max_memory)