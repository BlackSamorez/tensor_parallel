class SlicingConfig():
    def __init__(self, tensor_rules: dict, module_rules: dict):
        self.tensor_rules = tensor_rules
        self.module_rules = module_rules