from importlib.metadata import version


def verify_peft_version():
    peft_version = version("peft")
    if peft_version < "0.3.0":
        raise ImportError("tensor_parallel only works with peft>=0.3.0")
