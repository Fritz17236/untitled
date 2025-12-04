"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


import numpy as np
from numpy.typing import NDArray
from .params import RegressionParams


class NonlinearRegressorModel:
    def __init__(self, config: RegressionParams) -> None:
        self.config = config
        
        # instantiate neurons associated with this model  
        neurons = np.random.normal(loc=0, scale=1, size=(self.config.input_dimension, self.config.width))
        neurons = neurons / np.linalg.norm(neurons, axis=0)
        neurons = np.asarray(neurons)
        self.neurons = neurons
        
        self.decoders = None
        
    def fit(self, input_x: NDArray[np.floating], output_y: NDArray[np.floating]) -> float:
        """Fit the model to map the provided input to the provided output, reporting the resulting residual.

        Args:
            input_x (NDArray[np.floating]): Input having shape [NUM]
            output_y (NDArray[np.floating]): _description_
        """
        ...
        