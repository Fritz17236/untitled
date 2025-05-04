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
import os
import tqdm
import numpy as np
from torch.optim import LBFGS, AdamW
import torch
from numpy.typing import NDArray
from . import params


class RegressorType(Enum):
    HUBER = None

# TODO: gradient-based option if we cannot fit in memory or have existing decoder array to use  
def _regress_neuron(
    acts: NDArray[np.floating], target_outputs: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Regress a given neuron's activations onto the provided target output.

    Args:
        acts (NDArray[np.floating]): The neuron's activations for the provied sample data, having shape [NUM_SAMPLES] x [DEPTH]
        target_outputs (NDArray[np.floating]): The target outputs the neuron should be mapped to having shape [NUM_SAMPLES] x [OUTPUT_DIM]

    Returns:
        NDArray[np.floating]: The regression coefficients that map the neuron's output to target output having shape [DEPTH] x [OUTPUT_DIM]
    """
    return np.linalg.lstsq(acts, target_outputs, rcond=None)[0]

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

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
               
def _grad_step_neuron(
        acts: NDArray[np.floating], target_outputs: NDArray[np.floating], decoders: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Regress a given neuron's activations onto the provided target output, but by doing gradient-based update(s) 

        Args:
            acts (NDArray[np.floating]): The neuron's activations for the provied sample data, having shape [NUM_SAMPLES] x [DEPTH]
            target_outputs (NDArray[np.floating]): The target outputs the neuron should be mapped to having shape [NUM_SAMPLES] x [OUTPUT_DIM]
            decoders (NDArray[np.floating]): The existing decoder array (regression coefficients) to update, having shape  [DEPTH] x [OUTPUT_DIMENSION]

        Returns:
            NDArray[np.floating]: The regression coefficients that map the neuron's output to target output having shape [DEPTH] x [OUTPUT_DIM]
        """
        # FUCK YEAH STORE THE QR FACTORIZATION!!
        LEARNING_RATE = .01
        MAX_NUM_ITERS = 100
        clip_val = 1
        for j in range(MAX_NUM_ITERS):      
            grad = 2 * acts.T @ (acts @ decoders - target_outputs)    
             
            grad.clip(min=-clip_val, max=clip_val)
            
            if np.any(np.isnan(grad)):
                break
            if np.abs(grad).max().max() * LEARNING_RATE < 1e-8:
                break
            decoders -= LEARNING_RATE * grad

            
        return decoders        
        # A = torch.tensor(acts) # [S  D]
        # B = torch.tensor(target_outputs).T # [Y S] --> [S Y]
        # X = torch.tensor(decoders, requires_grad=True)# [D Y ]
        
        # es = EarlyStopping(tolerance=5, min_delta=1e-3)
        # optimizer = AdamW(params=[X])
        # prev_loss = np.inf
        # loss = np.inf
        # # optimizer = LBFGS(params=[X])
        # for j in range(MAX_ITERS := 1000):
        #     optimizer.zero_grad()
        #     output = A @ X
        #     loss = loss_fn(output, B)
        #     loss.backward()
        #     optimizer.step()
            
        #     es(loss, prev_loss)
        #     if es.early_stop:
        #         break
        
        # return X.detach().numpy()
        
def compute_decoders(
    activations: NDArray[np.floating],
    target_output: NDArray[np.floating],
    config: params.RegressionParams,
) -> NDArray[np.floating]:
    """Compute the decoders that map activations associated with target input to the target output

    Args:
        activations (NDArray[np.floating]): Activations associated with target input having shape [NUM_SAMPLES] x [DEPTH] x [WIDTH]
        target_output (NDArray[np.floating]): The target output to regress against, having shape [NUM_SAMPLES] x [OUTPUT_DIMENSION]
        config (params.RegressionParams): The regression configuration specifying, width, and depth.

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

def newton_step_decoder(
    activations_batch: NDArray[np.floating], 
    target_output_batch: NDArray[np.floating], 
    decoders: NDArray[np.floating], 
    config: params.RegressionParams
) -> NDArray[np.floating]:
    """Update the provided decoders using a newton step 
    Args:
        activations_batch (NDArray[np.floating]): batched activations representing input, having shape [DEPTH] x [WIDTH_NEURON_BATCH] x [NUM_SAMPLES_BATCH] 
        target_output_batch (NDArray[np.floating]): The target output to regress against, having shape [OUTPUT_DIMENSION] x [NUM_SAMPLES_BATCH]
        decoders (NDArray[np.floating]): The existing decoder array (regression coefficients) to update, having shape  [DEPTH] x [OUTPUT_DIMENSION] x [WIDTH_NEURON_BATCH]
        config (params.RegressionParams): The regression configuration specifying, width, and depth.

    Returns:
        NDArray[np.floating]:  The updated decoder array (regression coefficients), having shape  [DEPTH] x [OUTPUT_DIMENSION] x [WIDTH_NEURON_BATCH]
    """
    
    SAMPLE_DIM = 0
    OUTPUT_DIM = 1 
    DEPTH_DIM = 1
    WIDTH_DIM = 2
    
    act_depth = activations_batch.shape[DEPTH_DIM]
    act_num_samples = activations_batch.shape[SAMPLE_DIM]
    act_width = activations_batch.shape[WIDTH_DIM]
    
    output_dim = target_output_batch.shape[OUTPUT_DIM]
    target_num_samples = target_output_batch.shape[SAMPLE_DIM]
    
    if not act_depth == config.depth:
        raise ValueError(
             f"Mismatch between activations_batch array with shape (NUM_SAMPLES={act_num_samples}, DEPTH={act_depth}, WIDTH={act_width}), and depth set in config={config.depth}"
        )

    if not output_dim == config.output_dimension:
        raise ValueError(
             f"Mismatch between target output array with shape (NUM_SAMPLES={target_num_samples}, OUTPUT_DIMENSION={output_dim}), and output dimension set in config={config.output_dimension}"
        )

    if not act_num_samples == target_num_samples:
        raise ValueError(
            f"Mismatch between activations_batch array with shape output array with shape (NUM_SAMPLES={act_num_samples}, DEPTH={act_depth}, WIDTH={act_width}), and"\
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
                    func=_grad_step_neuron,
                    iterable=tqdm.tqdm(
                        [
                            (
                                activations_batch[:, :, idx_neuron],
                                target_output_batch,
                                decoders[:, :, idx_neuron]
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

def fit_qr(A: NDArray[np.floating], B: NDArray[np.floating], config: params.RegressionParams, neuron_slice: slice):
    
    # we should have R, Q2, QTB already initialized raise error if not 
    
    #  case 1: we have no R, Q2, QTB, lets create them 
    # rows == 0: 
        # Q1, R1 = qr_factorize(A) # check Q1, R1 shape against config + input data 
        # Q2, Rtilde = qr factorize(R1) # check Q2, Rtilde
        # Q = Q1 @ Q2  # check shape 
        # QTB = Q.T @ B # check shape 
        
        # decoder = inv(Rtilde) @ QTB
    
    # case 2: we must have all of these quantities already computed, let's update them with new data 
    
    ...
    
def qr_initialize(A: NDArray[np.floating], B: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Initialize a batched (TSQR) factorization from provided activation and target output data.   

    Args:
        A (NDArray[np.floating]): Input matrix to factorize, having shape [NUM_SAMPLES] x [DEPTH]
        B (NDArray[np.floating]): Target output matrix, having shape [NUM_SAMPLES] x [OUTPUT_DIMENSION]

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: Tuple respectively containing Rtilde, Q2, and QTB values to be stored persistently and updated via TSQR algorithm. 
    """ 
    Q1, R1 = np.linalg.qr(A, mode='reduced')
    Q2, Rtilde = np.linalg.qr(R1, mode='reduced')
    Q = Q1 @ Q2 
    QTB = Q.T @ B
    return (Rtilde, Q2, QTB)