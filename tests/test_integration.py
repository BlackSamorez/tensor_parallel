from typing import Tuple

import pytest
import torch
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock, _expand_mask, _make_causal_mask, build_alibi_tensor

from tensor_parallel import Config, TensorParallel
from tensor_parallel.state_actions import Scale, Split


@pytest.mark.parametrize("custom_config", [True, False])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3, ("cpu",) * 4])
def test_tp_bloom_block(devices, custom_config):
    torch.manual_seed(0)

    bloom_config = BloomConfig.from_pretrained("bigscience/bloom-560m")
    bloom_config.torch_dtype = torch.float32
    block = BloomBlock(bloom_config)

    tp_config = None
    if custom_config:
        tp_config = Config(
            state_rules={
                ".*self_attention\.query_key_value\.(weight|bias)": Split(world_size=len(devices), dim=0),
                ".*self_attention\.dense\.(weight|bias)": Split(world_size=len(devices), dim=0),
                ".*mlp\.dense_h_to_4h\.(weight|bias)": Split(world_size=len(devices), dim=0),
                ".*mlp\.dense_4h_to_h\.weight": Split(world_size=len(devices), dim=1),
                ".*mlp\.dense_4h_to_h\.bias": Scale(world_size=len(devices)),
            },
            input_rules={},
            output_rules={
                ".*self_attention\.query_key_value": {0: "gather -1"},
                ".*self_attention\.dense": {0: "gather -1"},
                ".*mlp\.dense_4h_to_h$": {0: "sum"},
            },
            attr_rules={},
        )

    test_inputs1 = torch.randn(2, 3, 1024, requires_grad=True, device=devices[0])
    test_inputs2 = test_inputs1.detach().clone().requires_grad_(True)
    batch_size = test_inputs1.shape[0]
    head_dim = len(block.input_layernorm.weight) // block.num_heads
    prefix_length = 5

    attention_mask = torch.ones(test_inputs1.shape[0], test_inputs1.shape[1] + prefix_length)
    alibi = build_alibi_tensor(attention_mask, block.num_heads, dtype=test_inputs1.dtype)
    attention_mask = _prepare_attn_mask(attention_mask, test_inputs1.shape[:2], prefix_length)

    layer_past = (
        torch.randn(batch_size * block.num_heads, head_dim, prefix_length, device=devices[0]),
        torch.randn(batch_size * block.num_heads, prefix_length, head_dim, device=devices[0]),
    )

    grad_proj = torch.rand_like(test_inputs1)
    y_ref, cache_ref = block(
        test_inputs1, use_cache=True, layer_past=layer_past, alibi=alibi, attention_mask=attention_mask
    )
    y_ref.backward(grad_proj)

    block_tp = TensorParallel(block, devices, tensor_parallel_config=tp_config)
    y_ours, cache_ours = block_tp(
        test_inputs2, use_cache=True, layer_past=layer_past, alibi=alibi, attention_mask=attention_mask
    )
    y_ours.backward(grad_proj)

    torch.testing.assert_close(y_ours, y_ref, atol=1e-6, rtol=1e-05)
    torch.testing.assert_close(test_inputs1.grad, test_inputs2.grad, atol=1e-6, rtol=1e-05)
    torch.testing.assert_close(cache_ref[0], cache_ours[0], atol=1e-6, rtol=1e-05)
    torch.testing.assert_close(cache_ref[1], cache_ours[1], atol=1e-6, rtol=1e-05)


def _prepare_attn_mask(attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int):
    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    if src_length > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, device=device, past_key_values_length=past_key_values_length
        )

    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask
