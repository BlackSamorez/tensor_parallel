# petals_local_parallel
YSDA project


## Benchmarking tutorial

1. Open benchmark.py, put your model in the cycle. 
2. run from console with ```CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 benchmark.py```
3. There are command line arguments: --do_backward: bool, --num_iter: int, --batch_size: int, 
and their short versions: -d, -n, -b

CUDA_VISIBLE_DEVICES -- gpus, you are using
nproc_per_node       -- # of gpus/ processes

So, for example, to run a benchmark with no backward, 1 million iterations and batch_size of 1488 objects 
one can write: 

``` CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 benchmark.py -d 0 -n 1e6 -b 1488```

or ``` CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 benchmark.py --do_backward 0 --num_iter 1e6 -batch_size 1488```
