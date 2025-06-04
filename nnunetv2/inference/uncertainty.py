"""Utility functions for uncertainty quantification.

This module provides simple wrappers around :class:`nnUNetPredictor` for obtaining
prediction uncertainty via Monte Carlo Dropout, Deep Ensembling and test time
augmentation (TTA). The functions return the mean of the predicted logits and
voxel-wise variances which can be used as measures of uncertainty.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import itertools
import torch
from torch.nn.modules.dropout import _DropoutNd

from .predict_from_raw_data import nnUNetPredictor


def _enable_dropout(model: torch.nn.Module) -> List[Tuple[_DropoutNd, bool]]:
    """Enable dropout layers during inference.

    Returns a list of tuples ``(layer, was_training)`` so that the original
    training state can be restored afterwards.
    """
    status = []
    for m in model.modules():
        if isinstance(m, _DropoutNd):
            status.append((m, m.training))
            m.train()
    return status


def _restore_dropout(status: List[Tuple[_DropoutNd, bool]]) -> None:
    for layer, was_training in status:
        layer.train(was_training)


def mc_dropout_prediction(
    predictor: nnUNetPredictor,
    data: torch.Tensor,
    num_samples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict with Monte Carlo dropout.

    Parameters
    ----------
    predictor: ``nnUNetPredictor``
        Initialized predictor.
    data: ``torch.Tensor``
        Preprocessed input tensor.
    num_samples: int
        Number of stochastic forward passes.

    Returns
    -------
    mean: ``torch.Tensor``
        Mean predicted logits.
    variance: ``torch.Tensor``
        Voxel-wise variance of the predictions.
    """
    predictor.network.eval()
    st = _enable_dropout(predictor.network)
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds.append(predictor.predict_logits_from_preprocessed_data(data))
    _restore_dropout(st)
    stack = torch.stack(preds)
    return stack.mean(0), stack.var(0)


def deep_ensemble_prediction(
    predictors: Iterable[nnUNetPredictor], data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate predictions from multiple predictors."""
    preds = []
    with torch.no_grad():
        for p in predictors:
            preds.append(p.predict_logits_from_preprocessed_data(data))
    stack = torch.stack(preds)
    return stack.mean(0), stack.var(0)


def tta_prediction(
    predictor: nnUNetPredictor,
    data: torch.Tensor,
    flips: Tuple[Tuple[int, ...], ...] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict with explicit test time augmentations.

    By default this will use all mirroring combinations of the axes specified in
    ``predictor.allowed_mirroring_axes``. ``predictor.use_mirroring`` is ignored
    and the network is called once per augmentation.
    """
    if flips is None:
        axes = predictor.allowed_mirroring_axes or ()
        flips = tuple(
            c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)
        )

    preds = []
    predictor.network.eval()
    with torch.no_grad():
        preds.append(predictor.predict_logits_from_preprocessed_data(data))
        for f in flips:
            flipped = torch.flip(data, dims=[i + 2 for i in f])
            pred = predictor.predict_logits_from_preprocessed_data(flipped)
            preds.append(torch.flip(pred, dims=[i + 2 for i in f]))
    stack = torch.stack(preds)
    return stack.mean(0), stack.var(0)

