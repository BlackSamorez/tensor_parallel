import torch
import threading


class AllReduce:
    def __init__(self, world_size: int, reduce_op: callable = sum):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op=torch.cat)
    
    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).reshape_as(x)


class ScatterReduce:
    def __init__(self, world_size: int, reduce_op: callable = sum):
        self.world_size = world_size
        self.tensor_parts = [[] for _ in range(world_size)]
        self.parts_ready = [threading.Event() for _ in range(world_size)]
        self.reduce_op = reduce_op
        
    def __call__(self, x: torch.Tensor, rank: int):
        try:
            for i, part in enumerate(x.flatten().tensor_split(self.world_size)):
                self.tensor_parts[i].append(part) # append is thread-safe. thanks, GIL!
                if len(self.tensor_parts[i]) == self.world_size:
                    self.parts_ready[i].set() # can be called more than once; we dont care

            self.parts_ready[rank].wait()
            parts_to_reduce = [part.to(x.device, non_blocking=True)
                               for part in self.tensor_parts[rank]]
            reduced_part = self.reduce_op(parts_to_reduce)
            return reduced_part
        finally:
            # prepare for next forward; each rank clears its own data
            self.tensor_parts[rank].clear()
            self.parts_ready[rank].clear()


class AllGather:
    def __init__(self, world_size: int, gather_op = torch.cat, ordered: bool = True):
        self.world_size = world_size
        self.parts = []
        self.parts_ready = threading.Event()
        self.gather_op = gather_op
        self.ordered = ordered
        
    def __call__(self, x: torch.Tensor, rank: int):
        try:
            parts, parts_ready = self.parts, self.parts_ready
            # ^-- note: we copy properties to locals so that the "finally" clause is thread-safe
            parts.append((rank, x))  # append is thread-safe. thanks, GIL!
            if len(parts) == self.world_size:
                parts_ready.set() # can be called more than once; we dont care
            parts_ready.wait()
            if self.ordered:
                parts = sorted(parts)  # sorted by rank
            parts = [part.to(x.device, non_blocking=True) for r, part in parts]
            # note: for one of the parts with r == rank, part.to(device) is a no-op
            return self.gather_op(parts)
        finally:
            self.parts = []
            self.parts_ready = threading.Event()
            # note: we can safely clear now because all ranks have copied 
            # self.parts_* to locals before passing parts_ready.wait
