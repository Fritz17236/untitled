"""basis_functions.py: Generate Nonliear Basis functions used for function approximation 
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy
import scipy.integrate
import traceback

def relu(x: np.ndarray) -> np.ndarray:
    """Elementwise Rectified Linear Unit 

    Args:
        x (np.ndarray): A numpy array to take ReLu of

    Returns:
        np.ndarray: A rectified linear unit transformation applied to the numpy array. 
    """
    if not np.all(np.isreal(x)):
        raise RuntimeError("X contained imaginary elements.")
    
    result = np.copy(x)
    result[result-0.5 < 0] = 0
    return result

def relu_callable(u: np.ufunc) -> np.ufunc:
    """Apply relu function to the callable

    Args:
        u (np.ufunc): Function to apply relu to

    Returns:
        np.ufunc: a the transformed function 
    """
    return lambda x: relu(x)
    
def inner_product_callable(f: np.ufunc, g: np.ufunc) -> float:
    """Compute the inner product of two continuous (discretely sampled) functions via numerical integration. 

     This is the inner product of two continuous functions, commonly written as <f, g>. 

    Args:
        f (np.ufunc): function in the first argument of the inner product
        g (np.ufunc): function in the second argument of the inner product
    Returns:
        float: the numerical inner product of the two functions sampled over the domain [0, 1]
    """
    # TODO: integration over vector-in, vector out continous functions 
    # prod, abs_err = scipy.integrate.quad(lambda x: f(x) * g(x), 0, 1 )
    domain = np.linspace(0,1, num=1000)
    prod = f(domain).T @ g(domain)
    return prod 

def inner_product(x: np.ndarray, y: np.ndarray) -> float:
    """Inner product of 2 numpy arrays

    Args:
        x (np.ndarray): array 1
        y (np.ndarray): array 2

    Returns:
        float: inner product
    """
    return x.T @ y 
    
def norm_l2(f: np.ufunc) -> float:
    """The norm of a function defined over the domain [0, 1]
    
    # TODO: integartion over vectorized domain, with vectorized codomain

    Args:
        f (np.ufunc): the function to compute the norm for

    Returns:
        float: the norm value of the function
    """
    return np.sqrt(inner_product_callable(f, f))

def proj(u: np.ufunc, v: np.ufunc) -> np.ufunc:
    """Compute the projection of function u onto function v 

    Args:
        u (np.ufunc): function to project onto
        v (np.ufunc): function to project

    Returns:
        np.ufunc: the projection of v onto u 
    """
    return lambda x: inner_product_callable(v, u) / inner_product_callable(u, u) * u(x)

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize an array by its l2 norm

    Args:
        x (np.ndarray): array to normalize

    Returns:
        np.ndarray: the normalized array
    """
    return x / np.sqrt(inner_product(x,x))
    
def normalize_callable(u: np.ufunc) -> np.ufunc:
    """Normalize a function by its L2 norm 

    Args:
        u (np.ufunc): Function to normalize

    Returns:
        np.ufunc: the normalized function 
    """
    norm = np.sqrt(inner_product_callable(u, u))
    return lambda x: u(x) / norm

def remove_basis_projection(u: np.ufunc, v: np.ufunc) -> np.ufunc:
    """Given a  basis u and function v, project function v onto basis u and remove this projection from v.
    
    In other words, remove the component of v that is parallel to the function u. 

    Args:
        u (np.ufunc): The basis function to remove via projection
        v (np.ufunc): The function we wish to remove basis u from. 

    Returns:
        np.ufunc: The function v with its "u" component removed.
    """
    return lambda x: v(x) - proj(u,v)(x)

def nonlinear_transform(u: np.ufunc, transform: np.ufunc = relu) -> np.ufunc:
    """Apply a (nonlinear) function to transform the input function 

    Args:
        u (np.ufunc): input function to transform
        transform (np.ufunc): the transformation to apply

    Returns:
        np.ufunc: the transformed function 
    """
    return lambda x: transform(u(x))

def gram_schmidt_loop(u: np.ufunc, transform: np.ufunc) -> np.ufunc:
    """Create a new basis function by the Gram-Schmidt process
    
    This process is summarized as 
    1) Apply a (nonlinear) transformation to a base function
    2) Remove the component of the base function from the transformed function via projection
    3) Normalize the resulting function to yield an orthonormal basis 

    Args:
        u (np.ufunc): base function 
        transform (np.ufunc): nonlinear transformation function

    Returns:    
        np.ufunc: a basis function that is orthonormal to u
    """
    try:
        _ = u(np.array([1, 2, 3]))
    except Exception as e:
        raise RuntimeError(f"The provided function {u} does not appear to be a numpy universal function:{traceback.format_exc()}")
    
    try:
        _ = transform(np.array([1, 2, 3]))
    except Exception as e:
        raise RuntimeError(f"The provided transform {transform} does not appear to be a numpy universal function:{traceback.format_exc()}")
    
    try:    
        return normalize_callable(remove_basis_projection(u, transform(u)))
    except Exception as e: 
        raise RuntimeError(f"\n\nUnhandled exception when performing gram-schmidt loop:\n\n {traceback.format_exc()}")

    
    

domain = np.linspace(0, 1, 1000)
f0 = domain
f0_norm = normalize_callable(f0)

plt.plot(domain, f0_norm(domain), label='f0(x)=x')
# plt.plot(domain,f1_norm(domain), label=f'f1=relu(f0(x)), area={norm}')

next_func = gram_schmidt_loop(f0, relu_callable)
plt.plot(domain, next_func(domain), label='1')


plt.plot(domain, gram_schmidt_loop(next_func, relu_callable)(domain), label='2')

plt.legend()
plt.show(block=True)