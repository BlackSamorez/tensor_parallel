from slicer_wrapper import SlicingConfig

SLICING_CONFIGS = {
"bigscience/bloom-560m": SlicingConfig(
    {
        "self_attention\.query_key_value": "vertical",
        "self_attention\.dense": "horizontal",
        "mlp\.dense_h_to_4h": "vertical",
        "mlp\.dense_4h_to_h": "horizontal",
    },
    {
        "self_attention": {"input": {"alibi": "cut 0"}, "output": {}, "attributes": {"num_heads": "scale_int"}},
        "self_attention\.dense": {"input": {}, "output": {"ALL": "reduce"}, "attributes": {}},
        "mlp\.dense_4h_to_h": {"input": {}, "output": {"ALL": "reduce"}, "attributes": {}},
    },
),

"bert-base-uncased": SlicingConfig(
    {
        "self\.query": "vertical",
        "self\.key": "vertical",
        "self\.value": "vertical",
        "attention\.output\.dense": "horizontal",
        "intermediate\.dense": "vertical",
        "[0-9]\.output\.dense": "horizontal",
    },
    {
        "attention\.self": {"input": {}, "output": {}, "attributes": {"num_attention_heads": "scale_int", "all_head_size": "scale_int"}},
        "attention\.output\.dense": {"input": {}, "output": {0: "reduce"}, "attributes": {}},
        "[0-9]\.output\.dense": {"input": {}, "output": {"ALL": "reduce"}, "attributes":{}},
    },
),

"t5-small": SlicingConfig(
    {
        "SelfAttention\.q": "vertical",
        "SelfAttention\.k": "vertical",
        "SelfAttention\.v": "vertical",
        "relative_attention_bias": "horizontal",
        "SelfAttention\.o": "horizontal",
        "DenseReluDense\.wi": "vertical",
        "DenseReluDense\.wi_0": "vertical",
        "DenseReluDense\.wi_1": "vertical",
        "DenseReluDense\.wo": "horizontal",
    },
    {
        "SelfAttention": {"input": {}, "output": {0: "reduce"}, "attributes": {"n_heads": "scale_int", "inner_dim": "scale_int"}},
        "relative_attention_bias": {"input": {}, "output": {}, "attributes": {"embedding_dim": "scale_int"}},
        "DenseReluDense": {"input": {}, "output": {"ALL": "reduce"}, "attributes": {}},
    },
)
}