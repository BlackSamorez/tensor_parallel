from slicer_wrapper import SlicingConfig

SLICING_CONFIGS = {
"bigscience/bloom-560m": SlicingConfig(
    {
        "self_attention.query_key_value": "vertical",
        "self_attention.dense": "horizontal",
        "mlp.dense_h_to_4h": "vertical",
        "mlp.dense_4h_to_h": "horizontal",
    },
    {
        "BloomAttention": {"input": {"alibi": "cut", 1: "scale"}, "output": {0: "reduce"}, "attributes": {"num_heads": "scale_int"}},
        "BloomMLP": {"input": {1: "scale"}, "output": {"ALL": "reduce"}, "attributes":{}},
    },
),

"bert-base-uncased": SlicingConfig(
    {
        "attention.self.query": "vertical",
        "attention.self.key": "vertical",
        "attention.self.value": "vertical",
        "attention.output.dense": "horizontal",
        "intermediate.dense": "vertical",
        "[0-9].output.dense": "horizontal",
    },
    {
        "BertSelfAttention": {"input": {}, "output": {}, "attributes": {"num_attention_heads": "scale_int", "all_head_size": "scale_int"}},
        "BertAttention": {"input": {}, "output": {0: "reduce"}, "attributes": {}},
        "BertOutput": {"input": {}, "output": {"ALL": "reduce"}, "attributes":{}},
    },
)
}