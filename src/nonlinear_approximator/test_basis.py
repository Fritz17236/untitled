"""test_basis.py: Unit testing code for basis_functions.py
"""


import numpy as np
from nonlinear_approximator.basis_functions import (
    relu,
    inner_product_callable,
)


def test_relu_1d():
    assert np.all(relu([1, 2, 3]) == np.array([1,2,3]))
    assert np.all(relu([-1, 2, 3]) == np.array([0,2,3]))
    assert np.all(relu([-1, 0, 3]) == np.array([0,0,3]))
    
    
def test_relu_2d():
    data_in = np.array([[1, 2, 3],[-1,2, 3], [-1, 0, 3]])
    data_out = np.array([[1,2,3], [0, 2, 3], [0, 0,3]])
    assert np.all(relu(data_in) == data_out)
    
    
def test_inner_product_1d():
    identity_func = lambda x: x
    assert np.isclose(inner_product_callable(identity_func, identity_func), 1/3)
    
def test_inner_product_2d():
    pass 

def test_norm():
    
        
        
        
