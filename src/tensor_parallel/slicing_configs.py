"""
Optimized configs for selected models
"""

from transformers import BloomConfig


def get_bloom_config(bloom_config: BloomConfig, devices: Sequence[torch.device]):
    return Config(
        state_rules={
            ".*self_attention\.query_key_value\.(weight|bias)": "split 0",
            ".*self_attention\.dense\.(weight|bias)": "split 0",
            ".*mlp\.dense_h_to_4h\.(weight|bias)": "split 0",
            ".*mlp\.dense_4h_to_h\.weight": "split 1",
            ".*mlp\.dense_4h_to_h\.bias": "scale",
        },
        input_rules={".*self_attention\.query_key_value": {"layer_past": func}},
        output_rules={
            ".*self_attention\.query_key_value": {0: "gather -1"},
            ".*self_attention\.dense": {0: "gather -1"},
            ".*mlp\.dense_4h_to_h$": {0: "sum"},
        },
        attr_rules={},
    )
