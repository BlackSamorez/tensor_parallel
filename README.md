# tensor_parallel

Run your PyTorch model on multiple GPUs from basic python

```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from tensor_parallel import tensor_parallel # <- interface for automatic optimal backend selection

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

model = tensor_parallel(model, ["cuda:0", "cuda:1"]) # <- magic happens here
# only half of the model is placed on each GPU reducing memory footprint twofold

inputs = tokenizer("Translate from German to English: How are you?", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs, num_beams=5)
print(tokenizer.decode(outputs[0]))  # Wie sind Sie?
```

## Installation

The recomended way to install this package is to use [pip](https://pypi.org/project/pip/):
```
pip install tensor_parallel
```

### Code style

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pycqa.github.io/isort/) for all pull requests.
Before committing your code, simply run `black . && isort .` and you will be fine.

--------------------------------------------------------------------------------
