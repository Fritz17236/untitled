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
import os 

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

    os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "9999s"
    os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "9999s"
    os.environ["DASK_DISTRIBUTED__DEPLOY__LOST_WORKER"] = "9999s"

    config = na.params.RegressionParams(
        width=100,
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


    model.fit(imgs_train[:, :], labels_train[:, :])

    # probs_train = model.predict(imgs_train[:, :])
    # preds_train = probs_train.argmax(axis=0)
    # acc_train = (sum(
    #     preds_train == (labels_train.argmax(axis=1)[:])
    #     ) / len(preds_train)).compute()
    
    # plt.hist(preds_train, bins=10)
    # plt.hist(labels_train.argmax(axis=1)[:].compute(), bins=10, alpha=0.3)
    # plt.title(f"Histogram of model predicted classifications on training data. Accuracy = {100 *acc_train}%")
    # plt.xlabel("Digit")
    # plt.ylabel(f"Number of classifciations (N={len(preds_train)})")
    # plt.show()
    