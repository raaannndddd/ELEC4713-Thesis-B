# models/base.py

class ModelClient:
    def __init__(self, model_name, temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt):
        raise NotImplementedError("Subclasses must implement generate()")