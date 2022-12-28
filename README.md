# tensor_parallel

Split your PyTorch model between multiple GPUs in one line of code. Doesn't require PhD in DeepSpeed configuration.

```python
import transformers
import tensor_parallel as tp
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])  # <- each GPU holds half of the weights

inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs, num_beams=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on my lap for a few minutes

model(input_ids=inputs, labels=inputs).loss.backward()  # train it normally
```


## Examples:

- [`examples/training_flan-t5-xl.ipynb`](./examples/training_flan-t5-xl.ipynb) - fine-tune full FLAN-T5 model on text summarization
- more examples TBA

## Installation
Latest stable version (recommended):
```
pip install tensor_parallel
```
Bleeding edge version:
```
pip install https://github.com/BlackSamorez/tensor_parallel/archive/main.zip
```

## FAQ

- __Q:__ I don't have a multi-GPU server. Can I use tensor_parallel in Google Colab?
- __A:__ Colab has a single GPU, but [Kaggle offers two T4 for free](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate) to all phone-verified accounts


- __Q:__ Should I use `TensorParallel` or `DataParallel`?
- __A:__ TensorParallel for large models, DataParallel for smaller ones


- __Q:__ How does it compare against FullyShardedDataParallel and ZeRO?
- __A:__ ZeRO is better if you can fit a large batch, TensorParallel is better for small batches

Why use `tensor_parallel` ...
- v.s. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
  - DeepSpeed has many parallelization strategies, but requires careful configuration
  - tensor_parallel has one strategy, but it works with 1 line of code
  - tensor_parallel works in a jupyter notebook
- v.s. [MegatronLM](https://github.com/NVIDIA/Megatron-LM)?
  - MegatronLM has _great_ tensor parallelism for one model architecture
  - tensor_parallel has _good_ parallelism for any architecture
  - tensor_parallel is way easier to install
- v.s. [parallelformers](https://github.com/tunib-ai/parallelformers)?
  - parallelformers implements a fixed [list of architectures](https://github.com/tunib-ai/parallelformers/tree/main/parallelformers/transformers)
  - tensor_parallel works for any architecture automatically 
  - parallelformers is inference-only, tensor_parallel supports training

In short, use

### Code style

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) and [isort](https://pycqa.github.io/isort/) for all pull requests.
Before committing your code, simply run `black . && isort .` and you will be fine.

--------------------------------------------------------------------------------
