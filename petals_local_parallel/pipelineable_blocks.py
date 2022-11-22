from typing import List, Tuple

import torch
from torch import nn

from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor, _make_causal_mask, _expand_mask
from transformers.models.bloom.configuration_bloom import BloomConfig

class MiddleBloom(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()

        self.num_heads = config.n_head
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
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

    def forward(self, hidden_states, attention_mask=None, past_key_values = None):
        batch_size, seq_length, _ = hidden_states.shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=0,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=False, output_attentions=False)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    None,
                    None,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=None,
                    use_cache=False,
                    output_attentions=False,
                    alibi=alibi,
                )

            hidden_states = outputs[0]

        return hidden_states
