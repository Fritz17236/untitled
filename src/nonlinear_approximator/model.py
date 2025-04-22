"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
import tqdm
from numpy.typing import NDArray
from .params import RegressionParams
from .activations import compute_activations
from .training import compute_decoders
from .inference import infer

class NonlinearRegressorModel:
    CONFIG_STRPATH = "configuration"
    DECODER_STRPATH = "decoders"
    def __init__(self, config: RegressionParams) -> None:
        self.config = config
        
        # instantiate neurons associated with this model  
        neurons = np.random.normal(loc=0, scale=1, size=(self.config.input_dimension, self.config.width))
        neurons = neurons / np.linalg.norm(neurons, axis=0)
        neurons = np.asarray(neurons)
        self.neurons = neurons # TODO: store these inside hdf5 database
        
        self.decoders = None
        
        
        if config.storage_path:
            self._h5py_save_path = config.storage_path
            self._init_hdf5()
        else:
            print("Storage path not configured, storing all model parameters in memory.")

    def fit(self, input_x: NDArray[np.floating], output_y: NDArray[np.floating]) -> None:
        """Fit the model to map the provided input to the provided output; TODO: report the resulting residual.

        Args:
            input_x (NDArray[np.floating]): Input having shape [NUM_INPUT_DIMS] x [NUM_SAMPLES]
            output_y (NDArray[np.floating]): Target output to fit having shape [NUM_OUTPUT_DIMS] x [NUM_SAMPLES]

        """
        if input_x.ndim != 2:
            raise ValueError(f"The provided input should have 2 dimensions (INPUT_DIM x NUM_SAMPLES), but had ndims={input_x.ndim} with shape {input_x.shape}")
        
        if output_y.ndim != 2:
            raise ValueError(f"The provided output should have 2 dimensions (OUTPUT_DIM x NUM_SAMPLES), but had ndims={output_y.ndim} with shape {output_y.shape}")

        if input_x.shape[1] != output_y.shape[1]: 
            raise ValueError(f"Mismatch between sample dimension (1) of input data ({input_x.shape[1]}, and target output ({output_y.shape[1]}))")
       
        if output_y.shape[0] != self.config.output_dimension:
           raise ValueError(f"Mismatch between provided output's dimension 0 size {output_y.shape[0]}, and configured output dimension {self.config.output_dimension}")
        
        if not self.config.storage_path: # in memory mode, hold all data (training data, neural decoders, and intermediate data) in memory
            activations_train = compute_activations(
                self.neurons,
                input_x=input_x,
                config=self.config
            )  
            
            self.decoders = compute_decoders(activations_train, output_y, self.config)   
            
        else: # use persistent storage
            self._check_storage_path_configured()
            with h5py.File(self.config.storage_path.resolve().absolute(), 'a') as file:
                
                # for batch in training data
                num_batches = np.ceil(input_x.shape[1] / self.config.batch_size)
                for batch in tqdm.tqdm(
                    np.array_split(input_x, num_batches, axis=1),
                    desc="Training on batches", 
                    total=num_batches,
                ):  
                    pbar = tqdm.tqdm(
                        total=self.config.width,
                        desc="Training across neurons...",
                        leave=False
                    )
                    
                    # for chunk in neural decoders:
                    for neuron_chunk in file[NonlinearRegressorModel.DECODER_STRPATH].iter_chunks():
                        
                        # get neural activations for this batch of data                         
                        slice_start = neuron_chunk[2].start
                        slice_end = self.config.width if not hasattr(neuron_chunk[2], 'end') else neuron_chunk[2].end
                        
                        activations = compute_activations(
                            self.neurons[:, slice_start:slice_end],
                            input_x=batch,
                            config=self.config
                        )  
                        
                        print("activations shpe: ", activations.shape)
                        #   load decoders in memory for this neuron chunk 
                        decoders = file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk]
                        if np.all(np.isnan(decoders)):
                            decoders[...] = compute_decoders(activations, output_y[:, slice_start:slice_end].T, self.config)   


                        # if we don't have decoders for this chunk
                        #   compute them for this batch, neuron chunk                        
                        # otherwise we have decoders:
                        #   update them with data from this batch, neuron chunk
                        
                        # save them to nvm for this neuron chunk        
                        
                        pbar.update(self.config.width // self.config.neuron_chunk_size)
                    pbar.close()
                ...
    
    def predict(self, input_x: NDArray[np.floating], average: bool=True) -> NDArray[np.floating]:
        
        if not self.config.storage_path: # in memory mode 
            if self.decoders is None:
                raise RuntimeError("The provided model has not been trained so cannot make a prediction. Call 'fit' first or 'load' first.")
            
            outputs = infer(input_x, self.neurons, self.decoders, self.config)
            
            if average: 
                return outputs.mean(axis=2).T
            else:
                return outputs
        else: # use persistent storage
            ...
    
    def save(self):
        """Save the model state to the path specified in its configuration"""
        with h5py.File(str(self.config.storage_path.resolve().absolute()), 'a') as file:
            file[NonlinearRegressorModel.CONFIG_STRPATH] = self.config
    
    def load(self):
        """Load the model state from the path specified in its configuration."""
        ...

    def clear_model(self):
        """" Clear any weights saved by the model """
        
        with h5py.File(self.config.storage_path.resolve().absolute(), 'w') as f:
            if NonlinearRegressorModel.DECODER_STRPATH in f.keys():
              del f[NonlinearRegressorModel.DECODER_STRPATH]
            
            
    def _init_hdf5(self):
        """ If a save path is configured, intialize the hdf5 data storage """
        self._check_storage_path_configured()
        with h5py.File(self.config.storage_path.resolve().absolute(), 'a') as file:
            print("Initializing HDF5 data store at ", self.config.storage_path.resolve().absolute())
            if NonlinearRegressorModel.CONFIG_STRPATH in file:
                del file[NonlinearRegressorModel.CONFIG_STRPATH]    
            file.create_dataset(NonlinearRegressorModel.CONFIG_STRPATH, data=self.config.model_dump_json())
    
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
        
    def _batch_fit():
        # TODO: batch across neurons, 
        # TODO: batch across samples, 
        ...   
        
    def _check_storage_path_configured(self): 
        """ Confirm that a save file exists for the configured hdf5 save path, raising a ValueError otherwise."""
        if not self._h5py_save_path:
            raise ValueError(f"There is no storage path configured: {self.config.storage_path=}")
        
    def _chunkify_by_neuron_axis(self, arr: NDArray[np.floating]) -> list[NDArray[np.floating]]:
        """Split an array into a list of subarrrays whose size is determined by the configuration.
        
        Any remainder of elements, smaller than the configured chunksize, is returned as the last element of the list.  

        Args:
            arr (NDArray[np.floating]): Array to split having shape [x] x [y] x [WIDTH]
            axis (int): Axis along which to split the array. 

        Returns:
            list[NDArray[np.floating]]: A list of arrays having length Ceiling([LENGTH_ALONG_AXIS] / [CONFIGURED_CHUNKSIZE]) each subarray having [CONFIGURED_CHUNKSIZE] elements, with the last having the modulo. 
        """
        if not self.config.neuron_chunk_size:
            raise RuntimeError("No chunk size was set in the configuration.")
        
        chunk_size = self.config.neuron_chunk_size
        if arr.shape[2] % chunk_size:
            return np.array_split(arr, arr.shape[2]//chunk_size + 1, axis=2)
        else:
            return np.array_split(arr, arr.shape[2]//chunk_size, axis=2)