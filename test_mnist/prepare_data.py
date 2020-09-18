import numpy as np

from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size_train = 64
batch_size_test = 128
learnrate = 0.01


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

#download
download = False
dataset_train = mnist.MNIST('../data/mnist', train=True,transform=transform, download=download )
dataset_test = mnist.MNIST('../data/mnist', train=False,transform=transform )

loader_tr = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
loader_ts = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)