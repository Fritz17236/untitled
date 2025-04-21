"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from .params import RegressionParams
from .activations import compute_activations
from .training import compute_decoders
from .inference import infer

class NonlinearRegressorModel:
    def __init__(self, config: RegressionParams) -> None:
        self.config = config
        
        # instantiate neurons associated with this model  
        neurons = np.random.normal(loc=0, scale=1, size=(self.config.input_dimension, self.config.width))
        neurons = neurons / np.linalg.norm(neurons, axis=0)
        neurons = np.asarray(neurons)
        self.neurons = neurons
        
        self.decoders = None
        
    def fit(self, input_x: NDArray[np.floating], output_y: NDArray[np.floating]) -> None:
        """Fit the model to map the provided input to the provided output; TODO: report the resulting residual.

        Args:
            input_x (NDArray[np.floating]): Input having shape [NUM_INPUT_DIMS] x [NUM_SAMPLES]
            output_y (NDArray[np.floating]): Target output to fit having shape [NUM_OUTPUT_DIMS] x [NUM_SAMPLES]

        """
        # Batch compute activations
        activations_train = compute_activations(
            self.neurons,
            input_x=input_x,
            config=self.config
        )  
        
        self.decoders = compute_decoders(activations_train, output_y, self.config)
    
    def predict(self, input_x: NDArray[np.floating], average: bool=True) -> NDArray[np.floating]:
        
        if self.decoders is None:
            raise RuntimeError("The provided model has not been trained so cannot make a prediction. Call 'fit' first or 'load' first.")
        
        outputs = infer(input_x, self.neurons, self.decoders, self.config)
        
        if average: 
            return outputs.mean(axis=2).T
        else:
            return outputs
    
    def save(self):
        """Save the model state to the path specified in its configuration"""
        # 
        ...
    
    def load(self):
        """Load the model state from the path specified in its configuration."""
        ...

    def _batch_fit():
        # TODO: batch across neurons, 
        # TODO: batch across samples, 
        ...   
        
        
# TODO:     
class DecoderWeights:
    """Data structure for storing neural decoder weights lazily on non-volatile memory (NVM) for quick saving/loading"""
# init - create hdf5 structure 
# save weights
# load weights
# info from config