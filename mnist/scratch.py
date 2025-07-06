import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import nonlinear_approximator as na
from importlib import reload
import torch
import torchvision
import PIL
from dask.distributed import Client, LocalCluster
import dask.delayed
import dask.array as da
from dask.diagnostics import ProgressBar

import dask.delayed
import dask.delayed


def one_hot_ten(int_label: torch.Tensor) -> NDArray[np.floating]:
    oh = np.zeros((10,))
    oh[int_label] = 1
    return oh

def to_numpy_arr(img: PIL.Image) -> NDArray[np.floating]:
    # rescale 255 to +/- 1 
    arr = np.asarray(img).flatten()
    arr = arr / 255 # 0 --> 1
    arr = arr - .5  # -.5 --> .5
    arr = 2 * arr   # -1 --> 1
    return arr

def load_data(): 
    train_data = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=to_numpy_arr, target_transform=one_hot_ten)
    test_data = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=to_numpy_arr, target_transform=one_hot_ten)

    imgs_train, labels_train = zip(*train_data)
    imgs_train = da.array(imgs_train)
    labels_train = da.array(labels_train)

    imgs_test, labels_test = zip(*test_data)
    imgs_test = da.array(imgs_test)
    labels_test = da.array(labels_test)
    
    return (
        imgs_train, labels_train,
        imgs_test, labels_test
    )


imgs_train, labels_train, imgs_test, labels_test = load_data()
print(f"Loaded training data images with shape {imgs_train.shape}, and labels with shape {labels_train.shape}")
print(f"Loaded test data images with shape {imgs_test.shape}, and labels with shape {labels_test.shape}")



def tent(x: NDArray[np.floating], params: na.params.TentParams) -> NDArray[np.floating]:
    """Apply the Tent Map either to a scalar or a vectorized array.
    https://en.wikipedia.org/wiki/Tent_map

    Arguments:
        x (float | NDArray[np.floating]): Input to apply transformation to, either a scalar, or applied elementwise to a vector
        params (TentParams): Configuration Containing:
            mu (float): The slope of the tent / triangle, typically between 0 and 2, with 2 being the upper boundary of bounded Chaotic behavior.
    Returns:
        float | NDArray[np.floating]: The mapped value or array, matching the type of data input.
    """
    return da.where(
        x > 0,
         params.mu * (1 - x),
         params.mu * (1 + x),
    )


def get_neurons(config): 
    neurons = da.random.normal(loc=0, scale=1, size=(config.input_dimension, config.width))
    neurons = neurons / da.linalg.norm(neurons, axis=0)
    return neurons


config = na.params.RegressionParams(
    width=1000,
    depth=50,
    input_dimension=imgs_train.shape[1],
    transform_type=na.activations.TransformType.TENT,
    transform_params=na.params.TentParams(mu=1.99),
    output_dimension=labels_train.shape[1],
    batch_size=1000,
)



neurons = get_neurons(config)
input_acts = da.matmul(imgs_train,  neurons)

acts = [input_acts]
for j in range(config.depth - 1):
    acts.append(
            tent(
                acts[-1],
                config.transform_params
            )
    )
acts = da.concatenate(acts, axis=-1)

if __name__ == '__main__': 
    with ProgressBar():
        # print(type(acts.compute().compute()))
        # decoders = da.linalg.lstsq(acts.rechunk({0: 'auto', 1:-1}), imgs_train)[0]
        decoders = da.linalg.lstsq(
                acts, 
                imgs_train
            )[0]
        decoders.to_zarr("decoder.zarr")