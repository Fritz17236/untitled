"""
training.py: Training Network Decoder Matrices from Activation Data
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from params import RegressionParams

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from sklearn.linear_model import HuberRegressor
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class RegressorType(Enum):
    HUBER = HuberRegressor


def compute_decoder(
    activations: NDArray[np.floating],
    target_output: NDArray[np.floating],
    config: RegressionParams,
    regressor: RegressorType = RegressorType.HUBER,
) -> NDArray[np.floating]:
    """Generate a decoder matrix by fitting a linear regression model mapping neural activations to desired function output.

    Args:
        activations (NDArray[np.floating]): Neural activation array having shape [DEPTH] x [WIDTH] x [NUM_SAMPLES]
        target_output (NDArray[np.floating]): Desired output functions having shape [OUPUT_DIMENSION] x [NUM_SAMPLES]
        config (RegressionParams): Parameters for regression, containing width and depth configuration for the network.
        regressor (RegressorType, optional): Regression function to use. Defaults to HuberRegressor.

    Returns:
        NDArray[np.floating]: _description_
    """

    # sanity check activations and target output match what is contained in config
    act_depth, act_width, act_num_samples = activations.shape
    output_dim, num_samples = target_output.shape

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

    # flatten according to configuration
    acts_flat = activations.reshape(
        (act_depth * act_width, num_samples), order=config.flatten_order.value
    )

    # fit huber model (or logic to match model enum)
    huber = regressor.value().fit(activations, target_output)

    # get coeffs and sanity check they are correct shape
    coeffs = huber.coef_
    dim0, dim1 = coeffs.shape
    if not dim0 == output_dim:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target output dimension = {output_dim}; dim0 of decoder matrix = {dim0}"
        )

    if not dim1 == act_depth * act_width:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target flattend depth*width dimension = {act_depth * act_width}; dim1 of decoder matrix = {dim1}"
        )

    return coeffs
