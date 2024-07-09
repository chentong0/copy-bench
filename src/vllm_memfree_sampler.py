from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os

import torch
import vllm.model_executor.layers.sampler as vllm_sampler
from transformers import AutoTokenizer
from vllm.model_executor.layers.ops.sample import sample as sample_triton
from vllm.model_executor.layers.sampler import Sampler as DefaultSampler
from vllm.model_executor.layers.sampler import (_apply_min_p,
                                                _apply_min_tokens_penalty,
                                                _apply_penalties,
                                                _apply_top_k_top_p,
                                                _build_sampler_output,
                                                _get_logprobs, _sample)
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors)
from vllm.sequence import SamplerOutput

# global variables for memfree decoding
memfree_tokenizer = None
memfree_prefix_to_rejection_token_id: Optional[Dict[Tuple, List]] = None
memfree_rejection_n_gram: Optional[int] = None

class Sampler(DefaultSampler):
    def __init__(self):
        super().__init__()
        if memfree_rejection_n_gram is not None:
            assert memfree_prefix_to_rejection_token_id is not None
            assert memfree_tokenizer is not None
            print(f"[INFO] Sampler initialized! n_gram={memfree_rejection_n_gram}, prefix set size={len(memfree_prefix_to_rejection_token_id)}")
    
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        # Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        # have not been generated yet
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_penalties, do_top_p_top_k,
        do_min_p) = SamplingTensors.from_sampling_metadata(
            sampling_metadata, vocab_size, logits.device, logits.dtype)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                    sampling_tensors.output_tokens,
                                    sampling_tensors.presence_penalties,
                                    sampling_tensors.frequency_penalties,
                                    sampling_tensors.repetition_penalties)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        if memfree_rejection_n_gram is not None:
            assert memfree_prefix_to_rejection_token_id is not None
            assert memfree_tokenizer is not None
            if sampling_tensors.output_tokens.shape[1] >= memfree_rejection_n_gram - 1:
                for i in range(logits.shape[0]):
                    output_ids = tuple(sampling_tensors.output_tokens[i].tolist())
                    vocab_size = logits.shape[-1]
                    output_ids = tuple([x for x in output_ids if 0 <= x < vocab_size])
                    n_1_gram = tuple(output_ids[-memfree_rejection_n_gram + 1:])
                    for reject_id in memfree_prefix_to_rejection_token_id.get(n_1_gram, []):
                        logits[i, reject_id] = -float("inf")
                        print(f'[INFO, pid={os.getpid()}] Rejected token {memfree_tokenizer.convert_ids_to_tokens([reject_id])[0]} for prefix {[memfree_tokenizer.convert_ids_to_tokens(x) for x in n_1_gram]} at position {sampling_tensors.output_tokens.shape[1]} of batch {i}/{logits.shape[0]}')

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            assert maybe_sampled_tokens_tensor is not None
            sampled_tokens_tensor = maybe_sampled_tokens_tensor
            on_device_tensors = (probs, sampled_tokens_tensor)
        else:
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results,
                                    sampling_metadata,
                                    prompt_logprobs,
                                    sample_logprobs,
                                    on_device_tensors=on_device_tensors)

def vllm_disable_memfree_decoding():
    # Monkey patch the original class with the custom class
    vllm_sampler.Sampler = DefaultSampler
    print(f'[INFO] patched vllm_sampler.Sampler with Default Sampler')
    # return DefaultSampler

def vllm_enable_memfree_decoding(model: str, rejection_texts: Optional[List[str]], rejection_n_gram = 10):

    def process(tokenizer, rejection_texts, rejection_n_gram):
        prefix_to_rejection_token_id = defaultdict(list)
        assert rejection_texts
        for text in rejection_texts:
            rejection_ids = tokenizer(text)["input_ids"]
            assert isinstance(rejection_ids, list)
            for i in range(rejection_n_gram - 1, len(rejection_ids)):
                n_1_gram = tuple(rejection_ids[i - rejection_n_gram + 1:i])
                prefix_to_rejection_token_id[n_1_gram].append(rejection_ids[i])
        for key, item in prefix_to_rejection_token_id.items():
            prefix_to_rejection_token_id[key] = sorted(set(item))
        return prefix_to_rejection_token_id

    global memfree_tokenizer, memfree_prefix_to_rejection_token_id, memfree_rejection_n_gram
    memfree_tokenizer = AutoTokenizer.from_pretrained(model)
    memfree_prefix_to_rejection_token_id = process(memfree_tokenizer, rejection_texts, rejection_n_gram)
    memfree_rejection_n_gram = rejection_n_gram
    
    # Monkey patch the original class with the custom class
    vllm_sampler.Sampler = Sampler
    print(f'[INFO] patched vllm_sampler.Sampler with MemFree Decoding Sampler, n_gram={rejection_n_gram}, prefix set size={len(memfree_prefix_to_rejection_token_id)}')
