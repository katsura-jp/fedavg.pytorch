import os
import os.path as ops
import urllib.request
import gzip
import numpy as np


def get_mnist_data(datadir):
    dataroot = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    os.makedirs(datadir, exist_ok=True)

    for key, filename in key_file.items():
        if ops.exists(ops.join(datadir, filename)):
            print(f"already downloaded : {filename}")
        else:
            urllib.request.urlretrieve(ops.join(dataroot, filename),
                                       ops.join(datadir, filename))

    with gzip.open(ops.join(datadir, key_file["train_img"]), "rb") as f:
        train_img = np.frombuffer(f.read(), np.uint8, offset=16)
    train_img = train_img.reshape(-1, 784)

    with gzip.open(ops.join(datadir, key_file["train_label"]), "rb") as f:
        train_label = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(ops.join(datadir, key_file["test_img"]), "rb") as f:
        test_img = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = test_img.reshape(-1, 784)

    with gzip.open(ops.join(datadir, key_file["test_label"]), "rb") as f:
        test_label = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_img, train_label, test_img, test_label
