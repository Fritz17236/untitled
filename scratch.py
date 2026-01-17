
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
from dask.diagnostics import ProgressBar
ProgressBar().register()

import torch
import tqdm
from scipy.stats import qmc
import dask.array as da

from sklearn.linear_model import Ridge
# %matplotlib inline
import numba
global USE_TORCH
USE_TORCH=False

@numba.jit(nopython=True)        
def xor(v) -> int:
    x, y = v[0], v[1]
    
    if (x < 0 and y > 0) or (x > 0 and y < 0):
        return 0 # return 0 if in 2nd or 4th quadrant 
    else:
        return 1 # return 1 if in 1st or 3rd quadrant

@numba.jit(nopython=True)            
def tent(x, mu=1.99):

    # return np.exp(-900 * np.square(x)) + -0.06045
    if USE_TORCH:
        # return mu * torch.minimum(x, 1-x)
        
        ...
    else:
        return mu * np.minimum(x, 1 - x)

@numba.jit(nopython=True)        
def compute_activations(neurons, input_x, depth, width):
    num_samples, dim_input = input_x.shape
    # neurons have dim_input x 3config.width shape
    if USE_TORCH:
        # activations = torch.zeros((num_samples, depth, width))
        # for idx_sample in range(num_samples):
        #     for idx_layer in range(depth):
        #         if idx_layer == 0:
        #             sample =  input_x[idx_sample, :]
        #             activations[idx_sample, idx_layer, :] = .5*(sample+1).T @ neurons
        #         else:
        #             activations[idx_sample, idx_layer, :] = tent(
        #                 activations[idx_sample, idx_layer - 1, :]
        #             )
        ...
    else:
        activations = np.zeros((num_samples, depth, width))
        for idx_sample in range(num_samples):
            for idx_layer in range(depth):
                if idx_layer == 0:
                    sample =  np.expand_dims(input_x[idx_sample, :], axis=-1)
                    # print(f"{input_x.shape=}, {activations.shape=}, {sample.shape=}, {neurons.shape=}")
                    activations[idx_sample, idx_layer, :] = .5*(sample+1).T @ neurons#1 - np.linalg.norm(neurons - sample,axis=0)/np.sqrt(dim_input)
                else:
                    activations[idx_sample, idx_layer, :] = tent(
                        activations[idx_sample, idx_layer - 1, :]
                    )
        
    return activations

@numba.jit(nopython=True)        
def flatten_activations(acts):
    num_samples, depth, width = acts.shape
    return acts.reshape((num_samples, depth * width ))

def gen_neurons(dimension, number):
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=number)
    if USE_TORCH: 
        # sample = torch.tensor(sample)
        ...
    return sample

# @numba.jit(nopython=False)        
def accum(input_x, output_y, neurons, depth, width, batch_size=5000):
    n_samples = input_x.shape[0]
    if USE_TORCH:
        # batch_xs = torch.split(input_x, n_samples // batch_size, dim=0)
        # batch_ys = torch.split(output_y, n_samples // batch_size, dim=0)  
        ...      
    else:
        batch_xs = np.array_split(input_x, n_samples // batch_size, axis=0)
        batch_ys = np.array_split(output_y, n_samples // batch_size, axis=0)
    DW = depth * width
    AtA = np.zeros((DW, DW))
    Atb = np.zeros((DW,1))
    for batch_x, batch_y in tqdm.tqdm(zip(batch_xs, batch_ys), total=len(batch_xs)):
        A_batch = flatten_activations(compute_activations(
            neurons, input_x=batch_x, depth=depth, width=width,
        ))
        AtA += A_batch.T @ A_batch
        Atb += A_batch.T @ batch_y
    return AtA, Atb



num_samples_train = 5000
num_samples_test = 1000

# Regression Params
config = params.RegressionParams(
    width=1000,
    depth=15,
    input_dimension=2,
    transform_type=activations.TransformType.TENT,
    transform_params= params.TentParams(mu=1.99),
    output_dimension=1,
) 

# Neurons drawn uniformly from surface of 2-sphere (unit circle)
# neurons = np.random.normal(loc=0, scale=1, size=(config.input_dimension, config.width))
# neurons = neurons / np.linalg.norm(neurons, axis=0)
# neurons = np.asarray(neurons)

neurons = gen_neurons(config.input_dimension, config.width).T
# Test + Train data drawn uniformly throughout 2-sphere (unit circle)
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

if USE_TORCH:
    test_samples = torch.tensor(test_samples)
    train_samples = torch.tensor(train_samples)

# Batch compute activations
activations_train = flatten_activations(compute_activations(
    neurons, input_x=train_samples.T, depth=config.depth, width=config.width,
))
activations_test = flatten_activations(compute_activations(
    neurons, input_x=test_samples.T, depth=config.depth, width=config.width,
))

# Compute XOR, then use masking to plot classification
xors_train = np.expand_dims(
    np.array([xor(train_samples[:, i]) for i in range(num_samples_train)], dtype=np.float64), axis=-1
)
xors_test = np.expand_dims(
    np.array([xor(test_samples[:, i]) for i in range(num_samples_test)], dtype=np.float64), axis=-1
)
if USE_TORCH:
    xors_train = torch.tensor(xors_train)
    xors_test = torch.tensor(xors_test)
    
mask_0 = (xors_train == 0).squeeze()
mask_1 = (xors_train == 1).squeeze()
plt.figure("xor demo")
plt.subplot(2,2,1)
plt.scatter(
    train_samples[0, mask_0],
    train_samples[1, mask_0],
    c="red",
    marker="x",
    label="XOR = 0",
)
plt.scatter(
    train_samples[0, mask_1],
    train_samples[1, mask_1],
    c="green",
    marker="o",
    label="XOR = 1",
)

plt.axis("equal")
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

plt.title("XOR Function")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()

# decoders = training.compute_decoders(activations_train, xors_train, config)
# preds = inference.infer()

AtA, Atb = accum(input_x=train_samples.T, output_y=xors_train, neurons=neurons, depth=config.depth, width=config.width )



AtA += np.eye(AtA.shape[0]) * 2e2

AtA_dask = da.from_array(AtA).rechunk({0: 'auto', 1: -1}).persist()
Atb_dask = da.from_array(Atb).rechunk({0: AtA_dask.chunksize[0], 1: -1}).persist()

dec = da.linalg.lstsq(AtA_dask, Atb_dask)[0].compute()

preds_train = activations_train @ dec
preds_test = activations_test @ dec
# preds_train = clf.predict(AtA)
# preds_train = clf.predict(activations_train)
# preds_test = clf.predict(activations_test)


xor_thresh = 0.5

xors_train_rounded = preds_train.copy()
xors_train_rounded[xors_train_rounded <= xor_thresh] = 0
xors_train_rounded[xors_train_rounded > xor_thresh] = 1
xor_train_actual = np.array(
    [xor(train_samples[:, i]) for i in range(train_samples.shape[1])]
)


mask_0 = np.isclose(xors_train_rounded, 0).squeeze()
mask_1 = np.isclose(xors_train_rounded, 1).squeeze()
plt.scatter(
    train_samples[0, mask_0],
    train_samples[1, mask_0],
    c="red",
    marker="x",
    label="XOR = 0",
)
plt.scatter(
    train_samples[0, mask_1],
    train_samples[1, mask_1],
    c="green",
    marker="o",
    label="XOR = 1",
)

mask_incorrect = (xors_train_rounded != xor_train_actual).squeeze()
plt.scatter(
    train_samples[0, mask_incorrect],
    train_samples[1, mask_incorrect],
    marker="+",
    c="yellow",
)

plt.axis("equal")
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

plt.title(
    f"Network Approximation of XOR function W={config.width}, D={config.depth}\nAccuracy = {100 * (1 - sum(mask_incorrect) / num_samples_train)}%"
)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()


xor_thresh = 0.5

xors_test_rounded = preds_test.copy()
xors_test_rounded[xors_test_rounded <= xor_thresh] = 0
xors_test_rounded[xors_test_rounded > xor_thresh] = 1
xors_test_actual = np.array(
    [xor(test_samples[:, i]) for i in range(test_samples.shape[1])]
)


mask_0 = np.isclose(xors_test_rounded, 0).squeeze()
mask_1 = np.isclose(xors_test_rounded, 1).squeeze()
plt.scatter(
    test_samples[0, mask_0],
    test_samples[1, mask_0],
    c="red",
    marker="x",
    label="XOR = 0",
)
plt.scatter(
    test_samples[0, mask_1],
    test_samples[1, mask_1],
    c="green",
    marker="o",
    label="XOR = 1",
)

mask_incorrect = (xors_test_rounded != xors_test_actual).squeeze()
plt.scatter(
    test_samples[0, mask_incorrect],
    test_samples[1, mask_incorrect],
    marker="+",
    c="yellow",
)

plt.axis("equal")
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")

plt.title(
    
    f"Network Approximation of XOR function W={config.width}, D={config.depth}\nAccuracy = {100 * (1 - sum(mask_incorrect) / num_samples_test):.2f}%"
)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()


