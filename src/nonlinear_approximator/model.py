"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

import numpy as np
import multiprocessing as mp 
import h5py
from pathlib import Path
from functools import cached_property
import tqdm
import os 
from zlib import adler32    
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
        
        if config.storage_path:
            self._init_hdf5()
        else:
            self._neurons = self._generate_neuron_directions()
            print("Storage path not configured, storing all model parameters in memory.")

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
       
        if not self.config.storage_path: # in memory mode, hold all data (training data, neural decoders, and intermediate data) in memory
            activations_train = compute_activations(
                self._neurons,
                input_x=input_x,
                config=self.config
            )  
            self.decoders = compute_decoders(activations_train, output_y, self.config)   
            
        else: # use persistent storage
            self._check_storage_path_configured()
            with h5py.File(self._h5py_save_path, 'a') as file:
                acts = compute_activations(self.neuron_directions, input_x, config=self.config)
                compute_decoders(acts, output_y, self.config ) # design consideration: have code only accept necessary params and not configuration 
                
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
            
        else: # use persistent storage
            self._check_storage_path_configured()
            with h5py.File(self._h5py_save_path, 'a') as file:
                output = infer(
                    input_x, 
                    self.neuron_directions, 
                    file[NonlinearRegressorModel.DECODER_STRPATH], 
                    self.config
                )
                if average:
                    return output.mean(axis=2)
    
    def save(self):
        """Save the model state to the path specified in its configuration"""
        with h5py.File(str(self._h5py_save_path), 'a') as file:
            if NonlinearRegressorModel.CONFIG_STRPATH not in file:
                raise RuntimeError(f"The provided model has no configuration stored, likely from a recent call to 'clear_model()'.")
            file[NonlinearRegressorModel.CONFIG_STRPATH] = self.config
    
    def load(self):
        """Load the model state from the path specified in its configuration."""
        ...

    def clear_model(self):
        """" Clear any data saved by the model """
        with h5py.File(self._h5py_save_path, 'w') as f:
            for key in f.keys():
                del f[key]
            euron
    @property
    def neuron_directions(self) -> NDArray[np.floating]:
        """ Fetch a copy of the model neurons' preferred directions.


        Returns:
            NDArray[np.floating]: The neurons used by the model to encode its input vector, having shape [INPUT_DIMENSION] x [WIDTH]
        """
        if hasattr(self, '_neurons'):
            return self._neurons
        else:
            self._check_storage_path_configured()
            with h5py.File(self._h5py_save_path, 'r') as f:
                return f[NonlinearRegressorModel.NEURON_STRPATH][:]

    @property
    def _h5py_save_path(self) -> Path:
        """The save path for the h5py file that model data will be stored in.
        
        Uses the hash of the configuration (json string) to track the config associated with the model. 
        
        """
        return self.config.storage_path.with_name(
            self.config.storage_path.stem 
            + f"_{adler32(self.config.model_dump_json().encode())}_"
            + self.config.storage_path.suffix
        )
             
    def _generate_neuron_directions(self) -> NDArray[np.floating]:
        """Generate neurons for the model according to its assigned coniguration 

        Returns:
            NDArray[np.floating]: An array of neurons having shape [INPUT_DIMENSION] x [WIDTH]
        """
        neurons = np.random.normal(loc=0, scale=1, size=(self.config.input_dimension, self.config.width))
        neurons = neurons / np.linalg.norm(neurons, axis=0)
        return np.asarray(neurons)
         
    def _init_hdf5(self):
        """ If a save path is configured, intialize the hdf5 data storage """
        self._check_storage_path_configured()
        
        with h5py.File(self._h5py_save_path.resolve().absolute(), 'a') as file:
            print("Initializing HDF5 data store at ", self._h5py_save_path)
            
            # Configuration of the model, dumped to json, and hash appended to the configured storage path.
            if NonlinearRegressorModel.CONFIG_STRPATH in file:
                if ( #if the model we're loading from is not the same as currently configured, raise an error.
                    # User should manually clear model to prevent accidental data loss. Not clear if we should ever hit this line anymore. 
                    adler32(str(file[NonlinearRegressorModel.CONFIG_STRPATH].asstr()[...]).encode()) 
                    !=
                    adler32(self.config.model_dump_json().encode())
                ):
                    raise RuntimeError("Tried to load model with a different configuration hash than the config provided to this model instance. ")  
            else:
             file[NonlinearRegressorModel.CONFIG_STRPATH] = self.config.model_dump_json()
    
            # if space for neural decoder matrices not allocated, do so. 
            if NonlinearRegressorModel.DECODER_STRPATH not in file:
                decoder_shape = (
                    self.config.depth,
                    self.config.output_dimension,
                    self.config.width
                )
                
                file.create_dataset(
                    name=NonlinearRegressorModel.DECODER_STRPATH, 
                    shape=decoder_shape, 
                    chunks=(self.config.depth, self.config.output_dimension, self.config.neuron_chunk_size),
                    dtype=np.floating,
                    fillvalue=np.nan         
                )
                
            # if neuron preferred directions are not present in file, create them. 
            if NonlinearRegressorModel.NEURON_STRPATH not in file:
                file.create_dataset(NonlinearRegressorModel.NEURON_STRPATH, data=self._generate_neuron_directions())
            
                
    def _check_storage_path_configured(self): 
        """ Confirm that a save file exists for the configured hdf5 save path, raising a ValueError otherwise."""
        if not self._h5py_save_path:
            raise ValueError(f"There is no stogit rage path configured: {self.config.storage_path=}")