import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import cm
from nonlinear_approximator import (
    activations,
    inference,
    model,
    params,
    training, 
)
import numba
import cProfile
from scipy.stats import qmc
from sklearn.linear_model import Ridge, RidgeCV, Lasso

# Parameters 
config1 = params.RegressionParams(
    width=2,
    depth=2,
    input_dimension=1,
    transform_type=activations.TransformType.TENT,
    transform_params= params.TentParams(mu=1.99),
    output_dimension=1,
)
num_samples_train = 1000
num_samples_test = 1000

def gen_neurons(dimension, number):
    # neurons = np.random.normal(loc=0, scale=1, size=(config.input_dimension, config.width))
    # neurons = np.random.uniform(low=-1, high=1, size=(config.input_dimension, config.width))
    # neurons = neurons / np.linalg.norm(neurons, axis=0)
    # neurons = np.asarray(neurons)    
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=number) 
    
    return sample
def rmse(x, y):
    return np.mean(np.abs(x.squeeze()-y.squeeze()))
    return np.sqrt(np.mean(np.square(x - y)))
    
@numba.jit(nopython=True)
def tent(x, mu=1.99):
    nonlinear_params = {"alpha": 900, "beta": -0.06045, "r": 3.99, "mu": 1.99}

    # return np.exp(-900 * np.square(x)) + -0.06045
    return mu * np.minimum(x, 1 - x)

def step(x):
    out = np.zeros_like(x)
    out[x > 0] = 1
    out[x == 0] = .5
    return out


@numba.jit(nopython=True)
def compute_activations(neurons, input_x, depth, width):
    num_samples, dim_input = input_x.shape
    # neurons have dim_input x config.width shape
    activations = np.zeros((num_samples, depth, width))
    for idx_sample in range(num_samples):
        for idx_layer in range(depth):
            if idx_layer == 0:
                # dots = neurons.T @ input_x[idx_sample, :]
                # activations[idx_sample, idx_layer, :] = np.where(dots > 0, dots, 0)
                sample =  np.expand_dims(input_x[idx_sample, :], axis=-1)
                activations[idx_sample, idx_layer, :] = .5*(sample+1).T @ neurons#1 - np.linalg.norm(neurons - sample,axis=0)/(2 * np.sqrt(dim_input))

            else:
                activations[idx_sample, idx_layer, :] = np.maximum(
                    activations[idx_sample, idx_layer - 1, :], 1 - activations[idx_sample, idx_layer - 1, :]
                )
    return activations
        
@numba.jit(nopython=True)
def accum(input_x, output_y, neurons, depth, width):
    batch_size = 256
    n_samples = input_x.shape[0]
    batch_xs = np.array_split(input_x, n_samples // batch_size, axis=0)
    batch_ys = np.array_split(output_y, n_samples // batch_size, axis=0)
    DW = depth * width
    AtA = np.zeros((DW, DW))
    Atb = np.zeros((DW,1))
    for batch_x, batch_y in zip(batch_xs, batch_ys):
        A_batch = compute_activations(
            neurons, input_x=batch_x, depth=depth, width=width
        ).reshape((batch_x.shape[0], DW))
        AtA += A_batch.T @ A_batch
        print(batch_y.shape)
        Atb += A_batch.T @ batch_y

    return AtA, Atb

def xor(v) -> int:
    x, y = v[0], v[1]
    
    if (x < 0 and y > 0) or (x > 0 and y < 0):
        return 0 # return 0 if in 2nd or 4th quadrant 
    else:
        return 1 # return 1 if in 1st or 3rd quadrant
    
with cProfile.Profile() as pr:
    config = params.RegressionParams(
        width=100,
        depth=25,
        input_dimension=2,
        transform_type=activations.TransformType.TENT,
        transform_params= params.TentParams(mu=1.99),
        output_dimension=1,
    )
    test_samples = np.random.normal(
        loc=0, scale=1, size=(config.input_dimension, num_samples_test)
    )
    test_samples /= np.linalg.norm(test_samples, axis=0)
    train_samples = np.random.normal(
        loc=0, scale=1, size=(config.input_dimension, num_samples_train)
    )
    train_samples /= np.linalg.norm(train_samples, axis=0)

    # Scale by radii distributed according to d^th root (where d is dimension) to get uniform density
    test_radii = np.random.uniform(2 * np.finfo(float).eps, 1, size=num_samples_test)
    train_radii = np.random.uniform(2 * np.finfo(float).eps, 1, size=num_samples_train)
    test_samples *= np.sqrt(test_radii)
    train_samples *= np.sqrt(train_radii)

    # Compute XOR, then use masking to plot classification
    xors_train = np.expand_dims(
        np.array([xor(train_samples[:, i]) for i in range(num_samples_train)]), axis=-1
    )
    xors_test = np.expand_dims(
        np.array([xor(test_samples[:, i]) for i in range(num_samples_test)]), axis=-1
    )

    neurons = gen_neurons(config.input_dimension, config.width).T

    AtA, Atb = accum(input_x=train_samples, output_y=xors_train, neurons=neurons, depth=config.depth, width=config.width )
    clf = Ridge(alpha=1e-3, tol=1e-16)
    # clf = Lasso(alpha=1e-5)
    clf.fit(AtA, Atb)
    
    # zero mean input
    # acts_train = compute_activations(
    #     neurons=neurons, input_x=xs_train, depth=config.depth, width=config.width
    # )
    acts_test = compute_activations(
        neurons=neurons, input_x=xs_test, depth=config.depth, width=config.width
    )
    # acts_train_flat = acts_train.reshape((len(xs_train), config.depth * config.width))
    acts_test_flat = acts_test.reshape((num_samples_test), config.depth*config.width)

    clf = Ridge(alpha=1e-3, tol=1e-16)
    preds = clf.predict(acts_test_flat)
    preds_fit = clf.predict(acts_train_flat)

    

plt.legend()
plt.gcf().tight_layout()
plt.ion()
plt.show()

pr.print_stats(sort='cumtime')