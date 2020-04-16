import sys
sys.path.append('./')

import pytest
import torch

from src.models.mlp import MLP
from src.models.cnn import CNN

@pytest.mark.parametrize(
    ["input_shape", "in_features", "num_classes", "hidden_dim"],
    [
        ((2, 1, 28, 28), 784, 10, 200),
    ],
)
def test_mlp_forward(input_shape, in_features, num_classes, hidden_dim):
    x = torch.FloatTensor(*input_shape)
    model = MLP(in_features, num_classes, hidden_dim)
    logits = model(x)
    assert logits.shape == (input_shape[0], num_classes)

@pytest.mark.parametrize(
    ["input_shape", "in_features", "num_classes"],
    [
        ((2, 1, 28, 28), 1, 10),
    ],
)
def test_cnn_forward(input_shape, in_features, num_classes):
    x = torch.FloatTensor(*input_shape)
    model = CNN(in_features, num_classes)
    logits = model(x)
    assert logits.shape == (input_shape[0], num_classes)


if __name__ == "__main__":
    test_cnn_forward((2, 1, 28, 28), 1, 10)