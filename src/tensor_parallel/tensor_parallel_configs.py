from .slicer_wrapper import Config

PREDEFINED_CONFIGS = {
    "BloomModel": Config(
        state_rules={
            ".*word_embeddings\.weight": "split 0",
            ".*self_attention\.query_key_value\.(weight|bias)": "split 0",
            ".*self_attention\.dense\.weight": "split 1",
            ".*self_attention\.dense\.bias": "scale",
            ".*mlp\.dense_h_to_4h\.(weight|bias)": "split 0",
            ".*mlp\.dense_4h_to_h\.weight": "split 1",
            ".*mlp\.dense_4h_to_h\.bias": "scale",
        },
        input_rules={
            ".*self_attention$": {"alibi": "split 0"},
        },
        output_rules={
            ".*word_embeddings\.weight": {0: "gather -1"},
            ".*self_attention\.dense$": {0: "sum"},
            ".*mlp\.dense_4h_to_h$": {0: "sum"},
        },
        attr_rules={
            ".*self_attention$": {"num_heads": "scale_int"},
        },
    ),
    # "BertForMaskedLM": Config(
    #     {
    #         ".*self\.query\.(weight|bias)": "vertical",
    #         ".*self\.key\.(weight|bias)": "vertical",
    #         ".*self\.value\.(weight|bias)": "vertical",
    #         ".*attention\.output\.dense\.(weight|bias)": "horizontal",
    #         ".*intermediate\.dense\.(weight|bias)": "vertical",
    #         ".*[0-9]\.output\.dense\.(weight|bias)": "horizontal",
    #     },
    #     {
    #         ".*attention\.self$": {"input": {}, "output": {}, "attributes": {"num_attention_heads": "scale_int", "all_head_size": "scale_int"}},
    #         ".*attention\.output\.dense$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
    #         ".*[0-9]\.output\.dense$": {"input": {}, "output": {0: "sum"}, "attributes":{}},
    #     },
    # ),
    # "T5Model": Config(
    #     {
    #         ".*SelfAttention\.q\.(weight|bias)": "vertical",
    #         ".*SelfAttention\.k\.(weight|bias)": "vertical",
    #         ".*SelfAttention\.v\.(weight|bias)": "vertical",
    #         ".*relative_attention_bias\.weight": "horizontal",
    #         ".*SelfAttention\.o\.(weight|bias)": "horizontal",
    #         ".*DenseReluDense\.wi\.(weight|bias)": "vertical",
    #         ".*DenseReluDense\.wi_0\.(weight|bias)": "vertical",
    #         ".*DenseReluDense\.wi_1\.(weight|bias)": "vertical",
    #         ".*DenseReluDense\.wo\.(weight|bias)": "horizontal",
    #     },
    #     {
    #         ".*SelfAttention$": {"input": {}, "output": {0: "sum"}, "attributes": {"n_heads": "scale_int", "inner_dim": "scale_int"}},
    #         ".*relative_attention_bias$": {"input": {}, "output": {}, "attributes": {"embedding_dim": "scale_int"}},
    #         ".*DenseReluDense$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
    #     },
    # )
}
