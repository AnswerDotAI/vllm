"""Test Cross Layer Attention (CLA)

Run 

```
export CUDA_VISIBLE_DEVICES=<your GPU ID>
export HF_HOME=<to store test models>
export HF_TOKEN=<your token>
pytest tests/answerdotai/test_kv_cache_sharing.py
```
"""
from typing import Tuple

import pytest
import torch
import time

from tests.kernels.utils import override_backend_env_variable
from vllm.utils import Device
from vllm.sampling_params import SamplingParams
from vllm.attention.backends.abstract import SharedSelfAttentionType


def reset_attn_outputs_and_metadatas(model):
    "Reset attn outputs and attn metadatas."
    for layer in model.model.layers:
        if len(layer.self_attn.attn_outputs) > 0:
            layer.self_attn.attn_outputs = []
    if len(model.model.attn_metadatas) > 0:
        model.model.attn_metadatas = []
        
    
def assert_attn_outputs(model, kv_cache_dtype):
    "Attention outputs of layer 0 and 1 should be equal (close) since KV is shared and Q is fixed to 1s during debug mode."        
    # Attention outputs of layer 0 and layer 1 should be equal, Q is fixed to 1s during debug mode.
    attn_outputs_layer0 = torch.cat(model.model.layers[0].self_attn.attn_outputs)
    attn_outputs_layer1 = torch.cat(model.model.layers[1].self_attn.attn_outputs)
    if kv_cache_dtype == "fp8":
        # a bit of precision loss due quantization + dequantization.
        closeness = torch.isclose(attn_outputs_layer0, attn_outputs_layer1, atol=1e-2, rtol=1e-2).float().mean()
        assert closeness > 0.99, \
            f"Attention outputs of layer 0 and layer 1 are not close:\n" \
            f"Layer 0: {attn_outputs_layer0}\n" \
            f"Layer 1: {attn_outputs_layer1}\n" \
            f"closeness: {closeness}"
    else:
        assert torch.equal(attn_outputs_layer0, attn_outputs_layer1), \
            f"Attention outputs of layer 0 and layer 1 are not equal:\n" \
            f"Layer 0: {attn_outputs_layer0}\n" \
            f"Layer 1: {attn_outputs_layer1}"
        

MODEL_AND_TOKENIZER = [
    ("answerdotai/vllm-tests-kv-cache-sharing-2layers-llama", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
 ]


@pytest.mark.parametrize("model_and_tokenizer", MODEL_AND_TOKENIZER)
@pytest.mark.parametrize("backend", ["XFORMERS_CLA"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("use_v2_block_manager", [False, True], ids=["v1_block_manager", "v2_block_manager"])
@pytest.mark.parametrize("enable_prefix_caching", [False, True], ids=["no_prefix_caching", "prefix_caching"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"], ids=["kv_cache_auto", "kv_cache_fp8"])
@pytest.mark.parametrize("enforce_eager", [False, True], ids=["cuda_graph", "eager"])
def test_prefill(
    hf_runner,
    vllm_runner,
    example_prompts,
    model_and_tokenizer: Tuple[str, str],
    backend: str,
    dtype: str,
    use_v2_block_manager: bool,
    monkeypatch,
    enable_prefix_caching: bool,
    kv_cache_dtype: str,
    enforce_eager: bool,
) -> None:
    """
    Test prefill with KV cache sharing.
    """
    override_backend_env_variable(monkeypatch, backend)

    # with hf_runner(model, dtype=dtype) as hf_model:
    #     hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    model, tokenizer = model_and_tokenizer
    cached_position = 0
    cached_prompt = example_prompts[cached_position]
    with vllm_runner(
            model,
            tokenizer_name=tokenizer,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map={0:0, 1:0}, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=True, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:


        model = vllm_model.model.llm_engine.model_executor.driver_worker.model_runner.model
        scheduler = vllm_model.model.llm_engine.scheduler[0]

        # Just prefill.
        reset_attn_outputs_and_metadatas(model)
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens=1)
        vllm_outputs_text1 = vllm_outputs[0][1]
        
        # Before cache hit.
        if enable_prefix_caching:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == 0.0
        else:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == -1
        
        # Attention outputs of layer 0 and layer 1 should be equal, Q is fixed to 1s during debug mode.
        assert_attn_outputs(model, kv_cache_dtype)
        
        # 1 attention metadata for a single model forward pass
        attn_metadata = model.model.attn_metadatas[0]
        assert attn_metadata.decode_metadata is None and len(model.model.attn_metadatas) == 1
        expected = [SharedSelfAttentionType.PREFILL_KV_NEW.name, SharedSelfAttentionType.PREFILL_KV_SHARED.name] # [layer 0, layer 1]
        assert attn_metadata.prefill_metadata.shared_self_attention_types == expected

        # After cache hit.
        reset_attn_outputs_and_metadatas(model)
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=1)
        vllm_outputs_text2 = vllm_outputs[0][1]

        if enable_prefix_caching:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) > 0.0
        else:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == -1
            
        # Attention outputs of layer 0 and layer 1 should be equal, Q is fixed to 1s during debug mode.
        assert_attn_outputs(model, kv_cache_dtype)
        
        # 1 attention metadata for a single model forward pass
        attn_metadata = model.model.attn_metadatas[0]
        assert attn_metadata.decode_metadata is None and len(model.model.attn_metadatas) == 1
        if enable_prefix_caching:
            expected = [SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV.name, SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV_SHARED.name] # [layer 0, layer 1]
        else:
            expected = [SharedSelfAttentionType.PREFILL_KV_NEW.name, SharedSelfAttentionType.PREFILL_KV_SHARED.name] # [layer 0, layer 1]
        assert attn_metadata.prefill_metadata.shared_self_attention_types == expected      

        assert vllm_outputs_text1 == vllm_outputs_text2


@pytest.mark.parametrize("model_and_tokenizer", MODEL_AND_TOKENIZER)
@pytest.mark.parametrize("backend", ["XFORMERS_CLA"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("use_v2_block_manager", [False, True], ids=["v1_block_manager", "v2_block_manager"])
@pytest.mark.parametrize("enable_prefix_caching", [False, True], ids=["no_prefix_caching", "prefix_caching"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"], ids=["kv_cache_auto", "kv_cache_fp8"])
@pytest.mark.parametrize("enforce_eager", [False, True], ids=["cuda_graph", "eager"])
def test_prefill_and_decode(
    hf_runner,
    vllm_runner,
    example_prompts,
    model_and_tokenizer: Tuple[str, str],
    backend: str,
    dtype: str,
    use_v2_block_manager: bool,
    monkeypatch,
    enable_prefix_caching: bool,
    kv_cache_dtype: str,
    enforce_eager: bool,
) -> None:
    """
    Test prefill and decode with KV cache sharing.
    """
    override_backend_env_variable(monkeypatch, backend)

    # with hf_runner(model, dtype=dtype) as hf_model:
    #     hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    model, tokenizer = model_and_tokenizer
    cached_position = 0
    cached_prompt = example_prompts[cached_position]
    with vllm_runner(
            model,
            tokenizer_name=tokenizer,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map={0:0, 1:0}, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=True, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:


        model = vllm_model.model.llm_engine.model_executor.driver_worker.model_runner.model
        scheduler = vllm_model.model.llm_engine.scheduler[0]

        # 3 tokens decode.
        reset_attn_outputs_and_metadatas(model)
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens=3)
        vllm_outputs_text1 = vllm_outputs[0][1]
        
        # Before cache hit.
        if enable_prefix_caching:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == 0.0    
        else:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == -1
        
        # Attention outputs of layer 0 and layer 1 should be equal, Q is fixed to 1s during debug mode.
        assert_attn_outputs(model, kv_cache_dtype)

        # XFormersCLAMetadata
        #     NOTE: Any python object stored here is not update when it is
        #     cuda-graph replayed. If you have values that need to be changed
        #     dynamically, it should be stored in tensor. The tensor has to be
        #     updated from `CUDAGraphRunner.forward` API.
        # Metadata would not be updated in cuda graph mode.    
        if enforce_eager:
            # 3 attention metadata for a 3 model forward pass (1 prefill + 2 decode)
            assert len(model.model.attn_metadatas) == 3 
            # Prefill phase.
            for meta in model.model.attn_metadatas[:1]:
                assert meta.decode_metadata is None               
                expected = [SharedSelfAttentionType.PREFILL_KV_NEW.name, SharedSelfAttentionType.PREFILL_KV_SHARED.name] # [layer 0, layer 1]
                assert meta.prefill_metadata.shared_self_attention_types == expected
            # Decode phases.
            for meta in model.model.attn_metadatas[1:]:
                assert meta.prefill_metadata is None
                expected = [SharedSelfAttentionType.DECODE_KV_NEW.name, SharedSelfAttentionType.DECODE_KV_SHARED.name] # [layer 0, layer 1]
                assert meta.decode_metadata.shared_self_attention_types == expected

        # After cache hit.
        reset_attn_outputs_and_metadatas(model)
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=3)
        vllm_outputs_text2 = vllm_outputs[0][1]
        
        if enable_prefix_caching:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) > 0.0    
        else:
            assert scheduler.block_manager.get_prefix_cache_hit_rate(Device.GPU) == -1

        # Attention outputs of layer 0 and layer 1 should be equal, Q is fixed to 1s during debug mode.
        assert_attn_outputs(model, kv_cache_dtype)

        # XFormersCLAMetadata
        #     NOTE: Any python object stored here is not updated when it is
        #     cuda-graph replayed. If you have values that need to be changed
        #     dynamically, it should be stored in tensor. The tensor has to be
        #     updated from `CUDAGraphRunner.forward` API.
        # Metadata would not be updated in cuda graph mode.    
        if enforce_eager:
            # 3 attention metadata for a 3 model forward pass (1 prefill + 2 decode)
            assert len(model.model.attn_metadatas) == 3 
            # Prefill phase.
            for meta in model.model.attn_metadatas[:1]:
                assert meta.decode_metadata is None
                if enable_prefix_caching:
                    expected = [SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV.name, SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV_SHARED.name] # [layer 0, layer 1]
                else:
                    expected = [SharedSelfAttentionType.PREFILL_KV_NEW.name, SharedSelfAttentionType.PREFILL_KV_SHARED.name] # [layer 0, layer 1]
                assert meta.prefill_metadata.shared_self_attention_types == expected
            # Decode phases.
            for meta in model.model.attn_metadatas[1:]:
                assert meta.prefill_metadata is None
                expected = [SharedSelfAttentionType.DECODE_KV_NEW.name, SharedSelfAttentionType.DECODE_KV_SHARED.name] # [layer 0, layer 1]
                assert meta.decode_metadata.shared_self_attention_types == expected
            
        assert vllm_outputs_text1 == vllm_outputs_text2
        

@pytest.mark.parametrize("model_and_tokenizer", MODEL_AND_TOKENIZER)
@pytest.mark.parametrize("backend", ["XFORMERS_CLA"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("use_v2_block_manager", [True], ids=["v2_block_manager"])
@pytest.mark.parametrize("enable_prefix_caching", [False], ids=["no_prefix_caching"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"], ids=["kv_cache_auto", "kv_cache_fp8"])
@pytest.mark.parametrize("enforce_eager", [False, True], ids=["cuda_graph", "eager"])
def test_throughput(
    vllm_runner,
    example_prompts,
    model_and_tokenizer: Tuple[str, str],
    backend: str,
    dtype: str,
    use_v2_block_manager: bool,
    monkeypatch,
    enable_prefix_caching: bool,
    kv_cache_dtype: str,
    enforce_eager: bool,
) -> None:
    """
    Test if KV cache sharing implementation harms the throughput.
    """
    override_backend_env_variable(monkeypatch, backend)

    model_name, tokenizer_name = model_and_tokenizer
   
    def _benchmark_speed(vllm_model, example_prompts):
        # warmup.
        _ = vllm_model.model.generate([example_prompts[0]], sampling_params=SamplingParams(temperature=0.0, max_tokens=1))
    
        # Time-to-first-token (TTFT).
        start = time.time()
        _ = vllm_model.model.generate(example_prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=1))
        ttft = time.time() - start
        
        # Completion time,
        start = time.time()
        outputs = vllm_model.model.generate(example_prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=32))
        completion_time = time.time() - start
        
        # total input tokens
        total_input_tokens = sum([len(o.prompt_token_ids) for o in outputs])
        total_output_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
        
        prefill_tput = total_input_tokens / ttft
        decode_tput = total_output_tokens / (completion_time - ttft)
        return prefill_tput, decode_tput
        

    # Base.
    with vllm_runner(
            model_name,
            tokenizer_name=tokenizer_name,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map=None, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=False, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:
        base_prefill_tput, base_decode_tput = _benchmark_speed(vllm_model, example_prompts)
           
    # Test: KV Cache Sharing.
    with vllm_runner(
            model_name,
            tokenizer_name=tokenizer_name,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map={0:0, 1:0}, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=False, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:

        test_prefill_tput, test_decode_tput = _benchmark_speed(vllm_model, example_prompts)
    
       
    assert test_prefill_tput > base_prefill_tput, f"Prefill throughput is degraded by {((base_prefill_tput - test_prefill_tput) / base_prefill_tput) * 100:.2f}%: {test_prefill_tput:.2f} vs {base_prefill_tput:.2f}"
    assert test_decode_tput > base_decode_tput, f"Decode throughput is degraded by {((base_decode_tput - test_decode_tput) / base_decode_tput) * 100:.2f}%: {test_decode_tput:.2f} vs {base_decode_tput:.2f}"
        
        
@pytest.mark.parametrize("model_and_tokenizer", MODEL_AND_TOKENIZER)
@pytest.mark.parametrize("backend", ["XFORMERS_CLA"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("use_v2_block_manager", [False, True], ids=["v1_block_manager", "v2_block_manager"])
@pytest.mark.parametrize("enable_prefix_caching", [False, True], ids=["no_prefix_caching", "prefix_caching"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"], ids=["kv_cache_auto", "kv_cache_fp8"])
@pytest.mark.parametrize("enforce_eager", [False, True], ids=["cuda_graph", "eager"])
def test_kv_cache_allocation(
    vllm_runner,
    model_and_tokenizer: Tuple[str, str],
    backend: str,
    dtype: str,
    use_v2_block_manager: bool,
    monkeypatch,
    enable_prefix_caching: bool,
    kv_cache_dtype: str,
    enforce_eager: bool,
) -> None:
    """
    Test num blocks allocated increases with KV cache sharing.
    """
    override_backend_env_variable(monkeypatch, backend)

    model_name, tokenizer_name = model_and_tokenizer
    
    # Base.
    with vllm_runner(
            model_name,
            tokenizer_name=tokenizer_name,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map=None, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=False, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:

        base_num_gpu_blocks = vllm_model.model.llm_engine.cache_config.num_gpu_blocks
        base_num_cpu_blocks = vllm_model.model.llm_engine.cache_config.num_cpu_blocks

    # Test: KV Cache Sharing.
    with vllm_runner(
            model_name,
            tokenizer_name=tokenizer_name,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            use_v2_block_manager=use_v2_block_manager,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            kv_cache_map={0:0, 1:0}, # layer 0 shares kv cache with layer 1
            debug_kv_sharing=False, # set q hidden states to 1.0 and store states/metadata.
            gpu_memory_utilization=0.2
    ) as vllm_model:
        
        test_num_gpu_blocks = vllm_model.model.llm_engine.cache_config.num_gpu_blocks
        test_num_cpu_blocks = vllm_model.model.llm_engine.cache_config.num_cpu_blocks
        
    assert test_num_gpu_blocks >= 2*base_num_gpu_blocks
    assert test_num_cpu_blocks >= 2*base_num_cpu_blocks
        
        