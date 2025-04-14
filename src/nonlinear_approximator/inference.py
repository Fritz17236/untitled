"""
inference.py: Code for evaluating trained regression models.
"""

from __future__ import annotations

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from params import RegressionParams
from numpy.typing import NDArray
from activations import compute_activations


def evaluate(
    x: NDArray, neurons: NDArray, decoder: NDArray, config: RegressionParams
) -> NDArray:
    """Pass input through the network and compute the output from the provided deocder matrix.

    Args:
        x (NDArray): Input array with shape [INPUT_DIMENSION] x [NUM_SAMPLES]]
        neurons (NDArray): Neuron preferred directions having shape [INPUT_DIMENSION] x [WIDTH]
        decoder (NDArray): Decoder matrix with shape [OUTPUT_DIMENSION] x ([WIDTH] * [DEPTH]) mapping neuron activations to model output.
        config (RegressionParams): Parameters for regression, containing width and depth configuration for the network.

    Returns:
        NDArray: The model outputs having shape [OUTPUT_DIMENSION] x [NUM_SAMPLES]
    """
    # TODO: refactor sanity check dim match code and add here
    acts = compute_activations(neurons, x, config)
    acts_flat = acts.reshape(
        (config.depth * config.width, x.shape[1]), order=config.flatten_order.value
    )
    return decoder @ acts_flat
