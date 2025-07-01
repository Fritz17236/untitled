import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import nonlinear_approximator as na
from importlib import reload
import torch
import torchvision
import PIL
from dask.distributed import Client, LocalCluster
import dask
import dask.array as da

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
    
    
if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=to_numpy_arr, target_transform=one_hot_ten)
    test_data = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=to_numpy_arr, target_transform=one_hot_ten)

    config = na.params.RegressionParams(
        width=1000,
        depth=50,
        input_dimension=len(train_data[0][0]),
        transform_type=na.activations.TransformType.TENT,
        transform_params=na.params.TentParams(mu=1.99),
        output_dimension=len(train_data[0][1]),
        batch_size=1000,
    )

    model = na.model.NonlinearRegressorModel(config)
    imgs_train, labels_train = zip(*train_data)
    imgs_train = da.array(imgs_train).persist()
    labels_train = da.array(labels_train).persist()


    imgs_test, labels_test = zip(*test_data)
    imgs_test = da.array(imgs_test).persist()
    labels_test = da.array(labels_test).persist()

    print(f"Loaded training data images with shape {imgs_train.shape}, and labels with shape {labels_train.shape}")
    print(f"Loaded test data images with shape {imgs_test.shape}, and labels with shape {labels_test.shape}")

    model.fit(imgs_train[:1000, :], labels_train[:1000, :])

