"""
activations.py: Definitions for applying transformations to data to produce neural activations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from params import LogisticParams, GaussParams, TentParams, RegressionParams
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from enum import Enum
from functools import partial
import numpy as np
from numpy.typing import NDArray


def logistic(x: NDArray[np.floating], params: LogisticParams) -> NDArray[np.floating]:
    """Apply the logistic map either to a scalar or vectorized array.

    https://en.wikipedia.org/wiki/Logistic_map

    Args:
        x (float | NDArray[np.floating]): Input to apply transformation to, either a scalar, or applied elementwise to a vector
        params (LogisticParams): Configuration containing:
            r (float): "Reproduction" value for the logistic map, typically betwen 0 and 4, with 4 being  the upper boundary of Chaotic behavior.


    Returns:
        float | NDArray[np.floating][float]: The mapped value or array, matching the type of data input.
    """
    return params.r * x * (1 - x)


def gauss(x: NDArray[np.floating], params: GaussParams) -> NDArray[np.floating]:
    """Apply the Gaussian Map either to a scalar or a vectorized array.

    https://en.wikipedia.org/wiki/Gauss_iterated_map
    Attributes:
        x (float | NDArray[np.floating]): Input to apply transformation to, either a scalar, or applied elementwise to a vector
        params (GaussParams): Configuraiton containing:
            alpha (float): The value that scales inside the exponential.  term fo the Gaussian function.
            beta (float): The value that offsets outside the exponentail term of the Gaussian function

    Returns:
        float | NDArray[np.floating][float]: The mapped value or array, matching the type of data input.
    """
    return np.exp(-params.alpha * np.square(x)) + params.beta


def tent(x: NDArray[np.floating], params: TentParams) -> NDArray[np.floating]:
    """Apply the Tent Map either to a scalar or a vectorized array.
    https://en.wikipedia.org/wiki/Tent_map

    Arguments:
        x (float | NDArray[np.floating]): Input to apply transformation to, either a scalar, or applied elementwise to a vector
        params (TentParams): Configuration Containing:
            mu (float): The slope of the tent / triangle, typically between 0 and 2, with 2 being the upper boundary of bounded Chaotic behavior.
    Returns:
        float | NDArray[np.floating]: The mapped value or array, matching the type of data input.
    """
    out = params.mu * (1 + x)
    out[x > 0] = params.mu * (1 - x[x > 0])
    return out


class TransformType(Enum):
    LOGISTIC = partial(logistic)
    GAUSS = partial(gauss)
    TENT = partial(tent)


def compute_activations(
    neurons: NDArray[np.floating],
    input_x: NDArray[np.floating],
    config: RegressionParams,
) -> NDArray[np.floating]:
    """Pass the provided input through the provided neurons and return the activations of those neruons.
    # TODO: doctstring
    """
    # neurons have shape [num_dims, width]
    # input has shape [num_dims, num_samples_input]
    n_dims_neuron, width = neurons.shape
    n_dims_input, num_samples = input_x.shape

    # neurons input dimension must match input dimension
    if not n_dims_neuron == n_dims_input:
        raise ValueError(
            f"Mismatch between neuron dimension [axis 0]={n_dims_neuron} and input dimension [axis=0]={n_dims_input}"
        )

    # input dimensions must match that contained in the config
    if not n_dims_neuron == config.input_dimension:
        raise ValueError(
            f"Mismatch between neuron dimension [axis 0]={n_dims_neuron} and input dimension set in config={n_dims_input}"
        )

    # number of neurons (width) must match that contained in the config
    if not width == config.width:
        raise ValueError(
            f"Mismatch between number of neurons provided={width} and the number of neurons set in config={config.width}"
        )

    activations = np.zeros((config.depth, config.width, num_samples))

    transform = config.transform_type.value

    if not transform:
        raise RuntimeError(
            f"Transform '{config.transform_type.name}' could not be found."
        )

    for idx_layer in range(config.depth):
        if idx_layer == 0:
            activations[idx_layer, :, :] = neurons.T @ input_x
        else:
            activations[idx_layer, :, :] = transform(
                activations[idx_layer - 1, :, :], config.transform_params
            )
    return activations
