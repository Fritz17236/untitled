"""
training.py: Training Network Decoder Matrices from Activation Data
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from params import RegressionParams

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from enum import Enum
import multiprocessing as mp
import os
import tqdm
import numpy as np
from numpy.typing import NDArray
import params


class RegressorType(Enum):
    HUBER = None


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


def compute_decoders(
    activations: NDArray[np.floating],
    target_output: NDArray[np.floating],
    config: params.RegressionParams,
) -> NDArray[np.floating]:
    """Compute the decoders that map activations associated with target input to the target output

    Args:
        activations (NDArray[np.floating]): Activations associated with target input having shape [DEPTH] x [WIDTH] x [NUM_SAMPLES]
        target_output (NDArray[np.floating]): The target output to regress against, having shape [NUM_SAMPLES] x [OUTPUT_DIMENSION]
        config (params.RegressionParams): The regression configuration specifying, width, and depth.

    Returns:
        NDArray[np.floating]: Decoder matrix for each neuron having shape [DEPTH] x [OUTPUT_DIMENSION] x [WIDTH]
    """
    # sanity check activations and target output match what is contained in config
    act_depth, act_width, act_num_samples = activations.shape
    num_samples, output_dim = target_output.shape

    if not act_depth == config.depth:
        raise ValueError(
            f"Mismatch between provided activations with depth dimension (0) = {act_depth} and provided configuration depth = {config.depth}"
        )

    if not act_width == config.width:
        raise ValueError(
            f"Mismatch between provided activations with width dimension (1) = {act_width} and provided configuration width = {config.width}"
        )

    if not output_dim == config.output_dimension:
        raise ValueError(
            f"Mismatch between provided target output dimension with dimension (0) = {output_dim} and provided configuration's output dimension = {config.output_dimension}"
        )

    if not act_num_samples == num_samples:
        raise ValueError(
            f"Mismatch between provided activations with sample count dimension (2) = {act_depth} and target outputs with sample count {num_samples}"
        )

    # multiprocessing to perform regression on per-neuron basis
    cpu_count = mp.cpu_count()
    if len(os.sched_getaffinity(0)) < cpu_count:
        try:
            os.sched_setaffinity(0, range(cpu_count))
        except OSError:
            print("Could not set affinity")
    num_worker_procs = len(os.sched_getaffinity(0))

    with mp.Pool(num_worker_procs) as p:
        decoders = np.stack(
            *[
                p.starmap(
                    func=_regress_neuron,
                    iterable=tqdm.tqdm(
                        [
                            (
                                activations[:, idx_neuron, :].T,
                                target_output,
                            )
                            for idx_neuron in range(config.width)
                        ],
                        total=config.width,
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
