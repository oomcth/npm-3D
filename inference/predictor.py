import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional


class Predictor:
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "mps"
        if torch.mps.is_available() else "cpu",
        preprocessing=None,
        postprocessing=None,
    ):
        self.model = model
        self.device = device
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs):
        if self.preprocessing:
            inputs = self.preprocessing(inputs)

        with torch.no_grad():
            outputs = self.model.generate(inputs["points"], inputs['prompt'])

        if self.postprocessing:
            outputs = self.postprocessing(outputs)

        return outputs
