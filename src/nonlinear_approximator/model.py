"""
model.py: Top-level model for orchestrating instantiation, training, and inference logic.
"""
from __future__ import annotations
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

import numpy as np
import h5py
from pathlib import Path
import tqdm
from zlib import adler32    
from numpy.typing import NDArray
from .params import RegressionParams
from .activations import compute_activations
from .training import compute_decoders, newton_step_decoder
from .inference import infer

class NonlinearRegressorModel:
    CONFIG_STRPATH = "configuration"
    DECODER_STRPATH = "decoders"
    NEURON_STRPATH = "neurons"
    
    def __init__(self, config: RegressionParams) -> None:
        self.config = config
        
        # instantiate neurons associated with this model  
        

        self.decoders = None
        
        if config.storage_path:
            self._init_hdf5()
        else:
            self._neurons = self._generate_neurons()
            print("Storage path not configured, storing all model parameters in memory.")

    def fit(self, input_x: NDArray[np.floating], output_y: NDArray[np.floating]) -> None:
        """Fit the model to map the provided input to the provided output; TODO: report the resulting residual.

        Args:
            input_x (NDArray[np.floating]): Input having shape [NUM_SAMPLES] x [NUM_INPUT_DIMS]
            output_y (NDArray[np.floating]): Target output to fit having shape  [NUM_SAMPLES] x [NUM_OUTPUT_DIMS]

        """
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
            
            print(f"Passing activations with shape: {activations_train.shape}\nPassing output with shape {output_y.shape}")
            self.decoders = compute_decoders(activations_train, output_y, self.config)   
            
        else: # use persistent storage
            self._check_storage_path_configured()
            with h5py.File(self._h5py_save_path, 'a') as file:
                
                # for batch in training data
                
                num_batches = int(np.ceil(input_x.shape[0] / self.config.batch_size))
                batches_input =  np.array_split(input_x, num_batches, axis=0)   
                batches_output = np.array_split(output_y, num_batches, axis=0)          
                for idx_batch in tqdm.tqdm(
                    iterable=range(num_batches),
                    desc="Training on batches", 
                    total=num_batches,
                ):  
                    batch_input = batches_input[idx_batch]
                    batch_output = batches_output[idx_batch]
                    
                    # for chunk in neural decoders:
                    for neuron_chunk in  tqdm.tqdm(
                        iterable=file[NonlinearRegressorModel.DECODER_STRPATH].iter_chunks(),
                        total=np.ceil(self.config.width / self.config.neuron_chunk_size),
                        desc="Training across neurons...",
                    ):
                        
                        # get neural activations for this batch ofs data                         
                        slice_start = neuron_chunk[2].start
                        slice_end = neuron_chunk[2].stop if not hasattr(neuron_chunk[2], 'end') else neuron_chunk[2].end
                        
                        activations = compute_activations(
                            neurons=self._neurons[:, slice_start:slice_end],
                            input_x=batch_input,
                            config=self.config
                        )                      
                        
                        #   load decoders in memory for this neuron chunk 
                        decoders = file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk]
                        
                        # if we don't have decoders for this chunk compute them for this batch, neuron chunk                        
                        if np.all(np.isnan(decoders)):
                            print(f"Decoders at slice {neuron_chunk} contained all NaN values. Initializing decoders...")
                            decoders[:] = compute_decoders(activations, batch_output, self.config)   
                        
                        else:
                            decoders[:] =  newton_step_decoder(activations, batch_output, decoders, self.config)          
                        
                        assert not np.any(np.isnan(decoders))
                        file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk] = decoders
                        assert not np.any(np.isnan(
                                                   file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk]
                                                   ))
                  
    def predict(self, input_x: NDArray[np.floating], average: bool=True) -> NDArray[np.floating]:
        # TODO: validate input_x shape 
        if not self.config.storage_path: # in memory mode 
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
                input_dim, num_samples = input_x.shape
                # use a recursive average to accumulate outputs from multiple chunks of neurons so we don't load all into memory at once 
                output_avg = None
                count_avg = 0
                
                for neuron_chunk in file[NonlinearRegressorModel.DECODER_STRPATH].iter_chunks():
                    print(f"neuron chunk: {neuron_chunk}; count_avg: {count_avg}")            
                    # get neural activations for this batch of data                         
                    slice_start = neuron_chunk[2].start
                    slice_end = neuron_chunk[2].stop if not hasattr(neuron_chunk[2], 'end') else neuron_chunk[2].end
                    
                    if np.any(np.isnan(
                        file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk]
                    )):
                        raise ValueError(f"Recorded Decoders for slice {neuron_chunk} contained NaN values: {file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk]}")
                    
                    output = infer(
                        input_x, 
                        self._neurons[:, slice_start:slice_end], 
                        file[NonlinearRegressorModel.DECODER_STRPATH][neuron_chunk], 
                        self.config
                    ).mean(axis=2).T
                    
                    count = slice_end - slice_start + 1

                    if count_avg == 0:
                        output_avg = output
                        count_avg = count
                    
                    else:
                        output_avg = (output * count + output_avg * count_avg) / (count + count_avg)
                        count_avg += count 
                        
                return output_avg.T
    
    def save(self):
        """Save the model state to the path specified in its configuration"""
        with h5py.File(str(self._h5py_save_path), 'a') as file:
            file[NonlinearRegressorModel.CONFIG_STRPATH] = self.config
    
    def load(self):
        """Load the model state from the path specified in its configuration."""
        ...

    def clear_model(self):
        """" Clear any weights saved by the model """
        
        with h5py.File(self._h5py_save_path, 'w') as f:
            for key in f.keys():
                del f[key]
            
    @property
    def neurons(self) -> NDArray[np.floating]:
        """ Fetch a copy of the model neurons' preferred directions.


        Returns:
            NDArray[np.floating]: The neurons used by the model to encode its input vector, having shape [INPUT_DIMENSION] x [WIDTH]
        """
        neurons = self._neurons 
        if isinstance(neurons, type(np.array([1]))):
            return neurons
        else:
            self._check_storage_path_configured()
            with h5py.File(self.config.storage_path, 'r') as f:
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
        
    # qr facotrize  Rx = Q^Tb ==> x = inv(R) @ Q.T @ b
    # how to store qrs for each neuron? 
    # grab a neuron chunk:
        # get associated q,r matrices with that chunk 
        # if they don't exist, create them 
            # qr factorize np.linalg.qr (mode = complete gives full mxm and m x n)
        # otherwise load Qs with shape (MxM x WIDTH), Rs with shape (MxN x WIDTH)
        
    # to update: call update scipy.linalg.qr_insert(Q, R, u=(new rows), k=0, which=row,)
        # store in place 
    
    def _generate_neurons(self) -> NDArray[np.floating]:
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
            if NonlinearRegressorModel.CONFIG_STRPATH in file:
                if (
                    adler32(str(file[NonlinearRegressorModel.CONFIG_STRPATH].asstr()[...]).encode()) 
                    !=
                    adler32(self.config.model_dump_json().encode())
                ):
                    raise RuntimeError("Tried to load model with a different configuration hash than the config provided to this model instance. ")  
            else:
             file[NonlinearRegressorModel.CONFIG_STRPATH] = self.config.model_dump_json()
    
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
                
            if NonlinearRegressorModel.NEURON_STRPATH not in file:
                file.create_dataset(NonlinearRegressorModel.NEURON_STRPATH, data=self._generate_neurons())
            self._neurons = file[NonlinearRegressorModel.NEURON_STRPATH][:]
    
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