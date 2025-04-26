"""
inference.py: Code for evaluating trained regression models.
"""

from __future__ import annotations

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


import multiprocessing as mp
import os
import numpy as np
import tqdm
from .params import RegressionParams
from numpy.typing import NDArray
from .activations import compute_activations


def infer(
    input_x: NDArray[np.floating],
    neurons: NDArray[np.floating],
    decoders: NDArray[np.floating],
    config: RegressionParams,
) -> NDArray[np.floating]:
    """Pass input through the model and infer the output

    Args:
        input_x (NDArray[np.floating]): The input to pass through the model having shape [NUM_SAMPLES] x [INPUT_DIM]
        neurons (NDArray): Neuron preferred directions having shape [INPUT_DIMENSION] x [WIDTH]
        decoders (NDArray[np.floating): The model weights mapping activations to the target function having shape [DEPTH] x [OUTPUT_DIM] x [WIDTH]
        config (RegressionParams): Parameters for regression, containing width and depth configuration for the network.

    Returns:
        NDArray[np.floating]: The inferred model output having shape [OUTPUT_DIM] x [NUM_SAMPLES] x [WIDTH]
    """
    # compute activations of input,
    acts = compute_activations(neurons, input_x, config)
    width = neurons.shape[1]
    
    cpu_count = mp.cpu_count()
    if len(os.sched_getaffinity(0)) < cpu_count:
        try:
            os.sched_setaffinity(0, range(cpu_count))
        except OSError:
            print("Could not set affinity")
    num_worker_procs = len(os.sched_getaffinity(0))

    # workers that scale across neurons (1 worker per neuron at a given time
    # do inference, then produce result
    with mp.Pool(num_worker_procs) as p:
        neuron_outputs = np.stack(
            *[
                p.starmap(
                    func=np.matmul,
                    iterable=tqdm.tqdm(
                        [
                            (
                                acts[
                                    :, :, idx_neuron
                                ],  # has shape [NUM_SAMPLES] x [DEPTH]
                                decoders[
                                    :, :, idx_neuron
                                ]  # has shape [DEPTH] x [OUTPUT_DIM]
                            )
                            for idx_neuron in range(width)
                        ],
                        total=width,
                    ),
                )
            ],
            axis=-1,
        )

    # compute and accumulate the model output
    return neuron_outputs
