import torch

from keras.datasets import cifar10

print('importing data')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('formatting inputs')
x_train = torch.tensor(x_train, dtype=torch.float32)/255.
x_train = x_train.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
x_test = torch.tensor(x_test, dtype=torch.float32)/255.0
x_test = x_test.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)

print('formatting outputs')
y_train = torch.tensor([elem[0] for elem in y_train], dtype=torch.long)
y_test = torch.tensor([elem[0] for elem in y_test], dtype=torch.long)