# compute relu

# compute norm 

# compute dot 

# x is input norm
# y is output norm, i.e. norm of relu(x) - dot(relu(x), x) x / x.T @ x

import numpy as np
import matplotlib.pyplot as plt 

num_dims = 1000
num_vecs = 10000
rand_vecs = (np.random.rand(num_dims, num_vecs) * 2 - 1) 


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

norms = np.linalg.norm(rand_vecs,axis=0)
dots = np.multiply(rand_vecs, relu(rand_vecs)).sum(axis=0)

assert np.all(
    np.isclose(
        norms, np.sqrt(np.multiply(rand_vecs, rand_vecs).sum(axis=0))
    )
)

coeffs = np.divide(dots, norms)
data = relu(rand_vecs) - (coeffs * rand_vecs)


out_norms = np.linalg.norm(data, axis=0)
# equal_line = np.linspace(np.min(out_norms), np.max(out_norms), num=1000)
# plt.scatter(norms, out_norms)
# plt.xlabel("Input Norms")
# plt.ylabel("Output Norms")
# plt.plot(equal_line, equal_line)
# plt.axis('auto')
# plt.hist(dots, bins=20)
# print(out_norms.shape)
# plt.show()

def forward(_x, iter=0): 
    _y = np.copy(_x)    
    # _y =  np.cos(_y + 2 * np.pi * iter / 100) - (_y.T @ np.cos(_y + 2 * np.pi * iter / 100)) / (_y.T @ _y) * _y
    # _y =  np.sin(_y + 2 * np.pi * iter / 100) - (_y.T @ np.sin(_y + 2 * np.pi * iter / 100)) / (_y.T @ _y) * _y

    # _y = relu(y) - (relu(y).T @ y) / (y.T @ y.T) * y 
    # _y = sigmoid(y) - (sigmoid(y).T @ y) / (y.T @ y.T) * y
    _y = np.tanh(y) - (np.tanh(y).T @ y) / (y.T @ y.T) * y

    return _y 

def col_dot(matrix, col1, col2):
    return np.sum(np.multiply(matrix[:,col1], matrix[:,col2]))
     
def normalize(arr):
    return arr / np.sqrt(arr.T @ arr)

x = np.linspace(-np.pi, np.pi, num=10000)
y = np.copy(x)
# plt.plot(x, x)
num_bases = 100
data = np.zeros((len(x), num_bases))
data[:, 0] = x
data[:, 0] /= np.sqrt(col_dot(data, 0, 0))
plt.figure('a')
for k in range(1, num_bases):
    data[:, k] = forward(data[:,k-1], 0)
    for i in range(k-1):
        data[:, k] -= col_dot(data, i, k) / col_dot(data, i, i) * data[:, i]
    data[:, k] /= np.sqrt(col_dot(data, k, k))# np.sum(np.abs(data[:,k]))
    
    plt.plot(x, data[:,k])
dot_prods = data.T @ data
dot_prods[dot_prods > 0] = np.sqrt(dot_prods[dot_prods > 0]) 


plt.matshow(dot_prods)
plt.colorbar()
plt.show()