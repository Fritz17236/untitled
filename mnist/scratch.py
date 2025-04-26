import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import tqdm
import nonlinear_approximator as na
from importlib import reload
import torch
import torchvision
import PIL
from pathlib import Path

#region Load and Plot MNIST Dataset 
# Preprocessing 
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

# load data
train_data = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=to_numpy_arr, target_transform=one_hot_ten)
test_data = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=to_numpy_arr, target_transform=one_hot_ten)

# process loaded data into images + labels for training and testig (cast to numpy since returned as list)
imgs_train, labels_train = zip(*train_data)
imgs_train = np.asarray(imgs_train).T
labels_train = np.asarray(labels_train).T

imgs_test, labels_test = zip(*test_data)
imgs_test = np.asarray(imgs_test).T
labels_test = np.asarray(labels_test).T

print(f"Loaded training data images with shape {imgs_train.shape}, and labels with shape {labels_train.shape}")
print(f"Loaded test data images with shape {imgs_test.shape}, and labels with shape {labels_test.shape}")

num_rows = 2
num_cols = 5

# Plotting
# plt.ion()
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5*num_cols,2*num_rows))
# for i in range(num_rows*num_cols):
#     ax = axes[i//num_cols, i%num_cols]
#     ax.imshow(train_data[i][0].reshape((28, 28)), cmap='gray')
#     ax.set_title('Label: {}'.format(np.argmax(train_data[i][1])))
# plt.tight_layout()
# plt.show()

#endregion

#region Config and Instantiate Model 
config = na.params.RegressionParams(
    width=1000,
    depth=50,
    input_dimension=len(train_data[0][0]),
    transform_type=na.activations.TransformType.TENT,
    transform_params=na.params.TentParams(mu=1.99),
    output_dimension=len(train_data[0][1]),
    neuron_chunk_size=100,
    # storage_path=Path('mnist_data.hdf5'),
    batch_size=1000,
)
model = na.model.NonlinearRegressorModel(config)
#endregion  


#region Train and Evaluate 

model.fit(imgs_train[:, :1000], labels_train[:,:1000])
#endregion 