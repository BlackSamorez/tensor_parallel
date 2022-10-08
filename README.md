# petals_local_parallel
YSDA project


## Benchmarking tutorial

1. Open benchmark.py, put your model in the cycle. 
2. run from console with ```CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 benchmark.py```

CUDA_VISIBLE_DEVICES -- gpus, you are using
nproc_per_node       -- # of gpus/ processes
