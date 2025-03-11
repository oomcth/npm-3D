import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional


class Predictor:
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu",
        preprocessing=None,
        postprocessing=None,
    ):
        self.model = model
        self.device = device
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs, batch_size=None):
        if self.preprocessing:
            inputs = self.preprocessing(inputs)

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, device=self.device)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            if batch_size:
                outputs = []
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i+batch_size]
                    batch_output = self.model.generate(batch)
                    outputs.append(batch_output)
                outputs = torch.cat(outputs, dim=0)
            else:
                outputs = self.model(inputs)

        outputs = outputs.cpu()

        if self.postprocessing:
            outputs = self.postprocessing(outputs)

        return outputs

    def predict_proba(self, inputs, batch_size=None):
        logits = self.predict(inputs, batch_size)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()

    def predict_classes(self, inputs, batch_size=None):
        logits = self.predict(inputs, batch_size)
        _, predicted = torch.max(logits, 1)
        return predicted.numpy()
