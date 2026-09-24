import os
import certifi
from torchvision import datasets

os.environ["SSL_CERT_FILE"] = certifi.where()

dataset1 = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

dataset2 = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)