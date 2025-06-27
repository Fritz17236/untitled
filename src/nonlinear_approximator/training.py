"""
training.py: Training Network Decoder Matrices from Activation Data
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from enum import Enum
import multiprocessing as mp
import h5py
import os
import tqdm
import numpy as np
from torch.optim import LBFGS, AdamW
import torch
from numpy.typing import NDArray
from . import params
from . import model

class RegressorType(Enum):
    HUBER = None

# TODO: gradient-based option if we cannot fit in memory or have existing decoder array to use  
def _regress_neuron(
    acts: NDArray[np.floating], target_outputs: NDArray[np.floating], h5py_strpath: str | None = None, idx: int | None = None
) -> NDArray[np.floating]:
    """Regress a given neuron's activations onto the provided target output.

    Args:
        acts (NDArray[np.floating]): The neuron's activations for the provied sample data, having shape [NUM_SAMPLES] x [DEPTH]
        target_outputs (NDArray[np.floating]): The target outputs the neuron should be mapped to having shape [NUM_SAMPLES] x [OUTPUT_DIM]

    Returns:
        NDArray[np.floating]: The regression coefficients that map the neuron's output to target output having shape [DEPTH] x [OUTPUT_DIM]
    """
    if h5py_strpath:
        with h5py.File(h5py_strpath, 'a') as file:
            decoder_path = model.NonlinearRegressorModel.DECODER_STRPATH
            q, r  = np.linalg.qr(acts)
            file[decoder_path][:, :, idx] = np.linalg.inv(r) @ q.T @ target_outputs 
    else:
        q, r = np.linalg.qr(acts, mode='reduced')
        return np.linalg.inv(r) @ q.T @ target_outputs

def loss_fn(U, V): 
    return torch.square(
        torch.norm(
            U - V,
            p='fro'
        )
    )   
    
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        # TODO document if keeping 
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

def compute_decoders(
    activations: NDArray[np.floating],
    target_output: NDArray[np.floating],
    config: params.RegressionParams,
    h5py_strpath: str | None = None
) -> NDArray[np.floating]:
    """Compute the decoders that map activations associated with target input to the target output

    Args:
        activations (NDArray[np.floating]): Activations associated with target input having shape [NUM_SAMPLES] x [DEPTH] x [WIDTH]
        target_output (NDArray[np.floating]): The target output to regress against, having shape [NUM_SAMPLES] x [OUTPUT_DIMENSION]
        config (params.RegressionParams): The regression configuration specifying, width, and depth
    Returns:
        NDArray[np.floating]: Decoder matrix for each neuron having shape [DEPTH] x [OUTPUT_DIMENSION] x [WIDTH]
    """
    # sanity check activations and target output match what is contained in config
    
    SAMPLE_DIM = 0
    OUTPUT_DIM = 1 
    DEPTH_DIM = 1
    WIDTH_DIM = 2
    
    act_depth = activations.shape[DEPTH_DIM]
    act_num_samples = activations.shape[SAMPLE_DIM]
    act_width = activations.shape[WIDTH_DIM]
    
    output_dim = target_output.shape[OUTPUT_DIM]
    target_num_samples = target_output.shape[SAMPLE_DIM]
    
    if not act_depth == config.depth:
        raise ValueError(
             f"Mismatch between activations array with shape (NUM_SAMPLES={act_num_samples}, DEPTH={act_depth}, WIDTH={act_width}), and depth set in config={config.depth}"
        )

    if not output_dim == config.output_dimension:
        raise ValueError(
             f"Mismatch between target output array with shape (NUM_SAMPLES={target_num_samples}, OUTPUT_DIMENSION={output_dim}), and output dimension set in config={config.output_dimension}"
        )

    if not act_num_samples == target_num_samples:
        raise ValueError(
            f"Mismatch between activations array with shape output array with shape (NUM_SAMPLES={act_num_samples}, DEPTH={act_depth}, WIDTH={act_width}), and"\
            f" target output array with shape (NUM_SAMPLES={target_num_samples}, OUTPUT_DIMENSION={output_dim})"
        )

    # multiprocessing to perform regression on per-neuron basis
    cpu_count = mp.cpu_count()
    if len(os.sched_getaffinity(0)) < cpu_count:
        try:
            os.sched_setaffinity(0, range(cpu_count))
        except OSError:
            print("Could not set affinity")
    num_worker_procs = len(os.sched_getaffinity(0)) - 1 
    
    with mp.Pool(num_worker_procs) as p:
        decoders = np.stack(
            *[
                p.starmap(
                    func=_regress_neuron,
                    iterable=tqdm.tqdm(
                        [
                            (
                                activations[:, :, idx_neuron],
                                target_output,
                                h5py_strpath,
                                idx_neuron,
                            )
                            for idx_neuron in range(act_width)
                        ],
                        total=act_width,
                        desc="Computing decoders",
                        leave=False
                    ),
                )
            ],
            axis=-1,
        )

    # confirm correct shape output
    dim0, dim1, dim2 = decoders.shape
    if not dim0 == act_depth:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target depth dimension = {act_depth}; dim0 of decoder matrix = {dim0}"
        )

    if not dim1 == output_dim:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target output dimension = {output_dim}; dim1 of decoder matrix = {dim1}"
        )

    if not dim2 == act_width:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target width dimension = {act_width}; dim2 of decoder matrix = {dim2}"
        )

    return decoders