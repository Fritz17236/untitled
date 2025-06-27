"""
activations.py: Definitions for applying transformations to data to produce neural activations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import LogisticParams, GaussParams, TentParams, RegressionParams
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
    LOGISTIC = 1
    GAUSS = 2
    TENT = 3


def compute_activations(
    neurons: NDArray[np.floating],
    input_x: NDArray[np.floating],
    config: RegressionParams,
) -> NDArray[np.floating]:
    """Pass the provided input through the provided neurons and return the activations of those neurons.

    Args:
        neurons (NDArray[np.floating]): Neurons to compute activations for with shape [INPUT_DIMENSION] x [WIDTH]
        input_x (NDArray[np.floating]): Input to compute activation for having shape [NUM_SAMPLES] x [INPUT_DIMENSION]
        config (RegressionParams): Regression configuration parameters

    Returns:
        NDArray[np.floating]: Neuron activations having shape [NUM_SAMPLES] x [DEPTH] x [WIDTH]
    """
    # neurons have shape [num_dims, width]
    # input has shape [num_dims, num_samples_input]
    n_dims_neuron, width = neurons.shape
    num_samples, n_dims_input= input_x.shape

    # neurons input dimension must match input dimension
    if not n_dims_neuron == n_dims_input:
        raise ValueError(
            f"Mismatch between neuron array with shape (INPUT_DIMENSION={n_dims_neuron}, WIDTH={width}), and input_x with shape (NUM_SAMPLES={num_samples}, INPUT_DIMENSION={n_dims_input})."
        )

    # input dimensions must match that contained in the config
    if not n_dims_neuron == config.input_dimension:
        raise ValueError(
             f"Mismatch between neuron array with shape (INPUT_DIMENSION={n_dims_neuron}, and input dimension set in config={config.input_dimension}"
        )


    activations = np.zeros((num_samples, config.depth, width))

    available_transforms = {
        TransformType.GAUSS.value: gauss,
        TransformType.LOGISTIC.value: logistic,
        TransformType.TENT.value: tent, 
    }
    transform: callable = available_transforms.get(config.transform_type.value)
    
    if not transform:
        raise RuntimeError(
            f"Transform '{config.transform_type}' could not be found. Available transforms: {available_transforms}")

    # layers must be computed sequntially, but across a given layer can be done in parallel 
    for idx_layer in range(config.depth):
        if idx_layer == 0:
            activations[:, idx_layer, :] = (input_x @ neurons)
        else:
            activations[ :,idx_layer, :] = transform(
                activations[ :,idx_layer - 1, :], config.transform_params
            )
    return activations