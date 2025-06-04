import torch

from nnunetv2.inference.uncertainty import (
    mc_dropout_prediction,
    deep_ensemble_prediction,
    tta_prediction,
)


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(p=0.5)
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.conv(x)


class DummyPredictor:
    def __init__(self):
        self.network = DummyNet()
        self.allowed_mirroring_axes = (0, 1)

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)[0]


def make_data() -> torch.Tensor:
    return torch.rand(1, 1, 4, 4)


def test_mc_dropout():
    predictor = DummyPredictor()
    mean, var = mc_dropout_prediction(predictor, make_data(), num_samples=5)
    assert mean.shape == (1, 4, 4)
    assert var.shape == (1, 4, 4)
    assert torch.any(var > 0)


def test_deep_ensemble():
    predictors = [DummyPredictor() for _ in range(3)]
    mean, var = deep_ensemble_prediction(predictors, make_data())
    assert mean.shape == (1, 4, 4)
    assert var.shape == (1, 4, 4)


def test_tta():
    predictor = DummyPredictor()
    mean, var = tta_prediction(predictor, make_data(), flips=((0,),))
    assert mean.shape == (1, 4, 4)
    assert var.shape == (1, 4, 4)


