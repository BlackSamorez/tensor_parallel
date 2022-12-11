from slicer_wrapper import SlicingConfig

SLICING_CONFIGS = {
"BloomModel": SlicingConfig(
    {
        ".*self_attention\.query_key_value\.(weight|bias)": "vertical",
        ".*self_attention\.dense\.(weight|bias)": "horizontal",
        ".*mlp\.dense_h_to_4h\.(weight|bias)": "vertical",
        ".*mlp\.dense_4h_to_h\.(weight|bias)": "horizontal",
    },
    {
        ".*self_attention$": {"input": {"alibi": "cut 0"}, "output": {}, "attributes": {"num_heads": "scale_int"}},
        ".*self_attention\.dense$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
        ".*mlp\.dense_4h_to_h$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
    },
),

"BertModel": SlicingConfig(
    {
        ".*self\.query\.(weight|bias)": "vertical",
        ".*self\.key\.(weight|bias)": "vertical",
        ".*self\.value\.(weight|bias)": "vertical",
        ".*attention\.output\.dense\.(weight|bias)": "horizontal",
        ".*intermediate\.dense\.(weight|bias)": "vertical",
        ".*[0-9]\.output\.dense\.(weight|bias)": "horizontal",
    },
    {
        ".*attention\.self$": {"input": {}, "output": {}, "attributes": {"num_attention_heads": "scale_int", "all_head_size": "scale_int"}},
        ".*attention\.output\.dense$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
        ".*[0-9]\.output\.dense$": {"input": {}, "output": {0: "sum"}, "attributes":{}},
    },
),

"T5Model": SlicingConfig(
    {
        ".*SelfAttention\.q\.(weight|bias)": "vertical",
        ".*SelfAttention\.k\.(weight|bias)": "vertical",
        ".*SelfAttention\.v\.(weight|bias)": "vertical",
        ".*relative_attention_bias\.weight": "horizontal",
        ".*SelfAttention\.o\.(weight|bias)": "horizontal",
        ".*DenseReluDense\.wi\.(weight|bias)": "vertical",
        ".*DenseReluDense\.wi_0\.(weight|bias)": "vertical",
        ".*DenseReluDense\.wi_1\.(weight|bias)": "vertical",
        ".*DenseReluDense\.wo\.(weight|bias)": "horizontal",
    },
    {
        ".*SelfAttention$": {"input": {}, "output": {0: "sum"}, "attributes": {"n_heads": "scale_int", "inner_dim": "scale_int"}},
        ".*relative_attention_bias$": {"input": {}, "output": {}, "attributes": {"embedding_dim": "scale_int"}},
        ".*DenseReluDense$": {"input": {}, "output": {0: "sum"}, "attributes": {}},
    },
)
}