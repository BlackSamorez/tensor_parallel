"""
Utility functions that help you process nested dicts, tuples, lists and namedtuples.
Based on: https://stackoverflow.com/questions/49739102/python-nested-dictionary-comparison
"""

from inspect import getmodule
from itertools import chain
from typing import Mapping, Optional, Sequence

from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from tensor_parallel.imports import verify_peft_version


def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True

    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True

    else:
        return True


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, BaseModelOutputWithPastAndCrossAttentions):
        return BaseModelOutputWithPastAndCrossAttentions(
            {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
        )
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def nested_map(fn, *t):
    # Check arguments.
    if not t:
        raise ValueError("Expected 2+ arguments, got 1")
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = "Nested structure of %r and %r differs"
            raise ValueError(msg % (t[0], t[i]))

    # Map.
    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])


def find_tied_weight_aliases(
    module: nn.Module, destination: Optional[Mapping[int, Sequence[str]]] = None, prefix: Optional[str] = None
) -> Mapping[int, Sequence[str]]:
    if prefix is None:
        prefix = ""
    if destination is None:
        destination = {}

    for name, param in chain(module._parameters.items(), module._buffers.items()):
        if param is not None:
            if id(param) in destination:
                destination[id(param)].append(prefix + name)
            else:
                destination[id(param)] = [prefix + name]

    for name, submodule in module._modules.items():
        if submodule is not None:
            find_tied_weight_aliases(module=submodule, destination=destination, prefix=prefix + name + ".")

    return destination


def check_lora(module: nn.Module) -> bool:
    """Checks if module is lora Linear from a correct version of PEFT

    Args:
        module (nn.Module): module to check

    Returns:
        bool: result
    """
    definition_module = getmodule(module)
    if definition_module is not None and definition_module.__name__ == "peft.tuners.lora":
        verify_peft_version()
        return type(module).__name__ == "Linear"
    else:
        return False
