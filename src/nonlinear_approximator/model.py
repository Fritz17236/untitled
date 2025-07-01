"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

import dask.distributed
import numpy as np
import dask.array as da
import dask 
from functools import cached_property
from numpy.typing import NDArray
from .params import RegressionParams
from .activations import compute_activations
from .training import compute_decoders
from .inference import infer


# Add qr storage to init with provided shape 
# assume persistent storage mode users qr,
    # redo fit method to perform and store qr factorization 
    # redo predict method to load qr factorization 
    

class NonlinearRegressorModel:
    CONFIG_STRPATH = "configuration"
    DECODER_STRPATH = "decoders"
    DECODER_STRPATH_Q = DECODER_STRPATH + "/Q"
    DECODER_STRPATH_R = DECODER_STRPATH + "/R"
    NEURON_STRPATH = "neurons"
    DATA_STRPATH = "data"
    
    def __init__(self, config: RegressionParams) -> None:
        self.config = config 

        self.decoders = None
        self._dask_cluster = dask.distributed.LocalCluster()
        self._dask_client = dask.distributed.Client(self._dask_cluster)
        self._neurons = self._generate_neuron_directions()

    def fit(self, input_x: NDArray[np.floating], output_y: NDArray[np.floating]) -> None:
        """Fit the model to map the provided input to the provided output; TODO: report the resulting residual.

        Args:
            input_x (NDArray[np.floating]): Input having shape [NUM_SAMPLES] x [NUM_INPUT_DIMS]
            output_y (NDArray[np.floating]): Target output to fit having shape  [NUM_SAMPLES] x [NUM_OUTPUT_DIMS]

        """
        
        # Data input validation.
        SAMPLE_DIM = 0
        INPUT_DIM = 1
        OUTPUT_DIM = 1
        
        if not isinstance(input_x, da.Array):
            input_x = da.asarray(input_x)
        if input_x.ndim != 2:
            raise ValueError(f"The provided input should have 2 dimensions (NUM_SAMPLES x NUM_INPUT_DIMS), but had ndims={input_x.ndim} with shape {input_x.shape}")
        
        if output_y.ndim != 2:
            raise ValueError(f"The provided output should have 2 dimensions ( NUM_SAMPLES x NUM_OUTPUT_DIMS), but had ndims={output_y.ndim} with shape {output_y.shape}")

        if output_y.shape[OUTPUT_DIM] != self.config.output_dimension:
           raise ValueError(f"Mismatch between provided output's dimension {OUTPUT_DIM} size {output_y.shape[OUTPUT_DIM]}, and configured output dimension {self.config.output_dimension}")
        
        if input_x.shape[INPUT_DIM] != self.config.input_dimension:
            raise ValueError(f"Mismatch between provided input's dimension {INPUT_DIM} size {input_x.shape[INPUT_DIM]} and configured input dimension {self.config.input_dimension}")

        if input_x.shape[SAMPLE_DIM] != output_y.shape[SAMPLE_DIM]: 
            raise ValueError(f"Mismatch between sample dimension ({SAMPLE_DIM}) of input data ({input_x.shape[SAMPLE_DIM]}, and target output ({output_y.shape[SAMPLE_DIM]}))")
       
        activations_train = compute_activations(
            self._neurons,
            input_x=input_x,
            config=self.config
        )  
    
        self.decoders = compute_decoders(self, activations_train, output_y, self.config)   
        
                        
                  
    def predict(self, input_x: NDArray[np.floating], average: bool=True) -> NDArray[np.floating]:
        # TODO: validate input_x shape 
        
        if not self.config.storage_path: # in memory mode, infer directly 
            if self.decoders is None:
                raise RuntimeError("The provided model has not been trained so cannot make a prediction. Call 'fit' first or 'load' first.")
            
            outputs = infer(input_x, self._neurons, self.decoders, self.config)
            
            if average: 
                return outputs.mean(axis=2).T
            else:
                return outputs
   
    def save(self):
        """Save the model state to the path specified in its configuration"""
        ...    
    def load(self):
        """Load the model state from the path specified in its configuration."""
        ...

    def clear_model(self):
        """" Clear any data saved by the model """
        ...
            
    @cached_property
    def neuron_directions(self) -> NDArray[np.floating]:
        """ Fetch a copy of the model neurons' preferred directions.

        Returns:
            NDArray[np.floating]: The neurons used by the model to encode its input vector, having shape [INPUT_DIMENSION] x [WIDTH]
        """
        return self._neurons

    def _generate_neuron_directions(self) -> NDArray[np.floating]:
        """Generate neurons for the model according to its assigned coniguration 

        Returns:
            NDArray[np.floating]: An array of neurons having shape [INPUT_DIMENSION] x [WIDTH]
        """
        neurons = da.random.normal(loc=0, scale=1, size=(self.config.input_dimension, self.config.width))
        neurons = neurons / da.linalg.norm(neurons, axis=0)
        return da.array(neurons)
