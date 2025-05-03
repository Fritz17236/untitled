"""
params.py: Configuration Parameters for nonlinear manifold regression.
"""

from __future__ import annotations

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from .activations import TransformType


# Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array, and only refer to the order of indexing.
class FlattenOrder(Enum):
    """Read the elements of a using this index order, and place the elements into the reshaped array using this index order

    ‘C’ means to read / write the elements using C-like index order, with the last axis index changing fastest, back to the first axis index changing slowest.

    ‘F’ means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest.

    'A' means to read / write the elements in Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise.
    """

    C_STYLE = "C"
    FORTRAN_STYLE = "F"
    AUTO = "A"


class NonlinearTransformationParams(BaseModel, frozen=True):
    """Base Class for Nonlinear Parameters"""


class LogisticParams(NonlinearTransformationParams, frozen=True):
    """Parameters for the logistic  map nonlinear transformation
    https://en.wikipedia.org/wiki/Logistic_map
    Attributes:
        r (float): "Reproduction" value for the logistic map, typically betwen 0 and 4, with 4 being  the upper boundary of Chaotic behavior.
    """

    r: float


class GaussParams(NonlinearTransformationParams, frozen=True):
    """Parameters for the Gaussian Step Map
    https://en.wikipedia.org/wiki/Gauss_iterated_map
    Attributes:
        alpha (float): The value that scales inside the exponential.  term fo the Gaussian function.
        beta (float): The value that offsets outside the exponentail term of the Gaussian function
    """

    alpha: float
    beta: float


class TentParams(NonlinearTransformationParams, frozen=True):
    """Parameters for the Tent Map
    https://en.wikipedia.org/wiki/Tent_map
    Attributes:
        mu (float): The slope of the tent / triangle, typically between 0 and 2, with 2 being the upper boundary of bounded Chaotic behavior.
    """

    mu: float


class RegressionParams(BaseModel, frozen=True):
    """
    Parameters for a nonlinear regression network approximator.

    Attributes:
        width (int): The number of neurons in the network (i.e. the width of the network)
        depth (int): The number of layers of the network (i.e. the depth of the network)
        input_dimension (int): The dimension of the data vectors we input to the network
        output_dimension (int): The dimension of the data vectors we expect the network to output
        transform_type (TransformType): The type of nonlinear transformation to use between layers
        transform_params (NonlinearTransformationParams): The parameters for the nonlinear transformation.
        storage_path (Path): Path on the local filesystem to save and/or load model state (e.g. decoder weights) to/from 
        neuron_chunk_size (int): The number of neurons to consider at once when loading into memory, only applies if storage path is set
        batch_size (int): The number of samples to use when computing updates to existing decoder weights (currrently) only applies is storage path is set
    """
    # TODO: sanity check that batch size if not -1 is >= output dimension. 
    width: int
    depth: int
    input_dimension: int
    output_dimension: int
    transform_type: TransformType
    transform_params: NonlinearTransformationParams
    storage_path: Path | None = None
    neuron_chunk_size: int | None = 250
    batch_size: int | None = -1 