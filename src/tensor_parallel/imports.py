import pkg_resources


def verify_peft_version():
    peft_version = pkg_resources.get_distribution("peft").version
    if peft_version < "0.3.0":
        raise ImportError("tensor_parallel only works with peft>=0.3.0")
