"""
params.py: Configuration Parameters for nonlinear manifold regression.
"""
__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"

from pydantic import BaseModel
from enum import Enum

class TransformType(Enum):
    LOGISTIC = 1
    GAUSS = 2
    TENT = 3
    

class NonlinearTransformationParams(BaseModel, frozen=True):
    """Base Class for Nonlinear Parameters
    """
    pass

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
        transform_type (TransformType): The type of nonlinear transformation to use between layers
        transform_params (NonlinearTransformationParams): The parameters for the nonlinear transformation. 
         
    """
    width: int
    depth: int 
    input_dimension: int 
    transform_type: TransformType
    transform_params: NonlinearTransformationParams
    
    
    
