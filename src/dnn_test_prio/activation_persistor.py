"""Utility (not used for paper) to extract activations from our models"""
import os.path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.dnn_test_prio.handler_model import BaseModel

BADGE_SIZE = 100


def _persist_badge(
    case_study: str,
    model_id: int,
    dataset: str,
    badge_id: int,
    activations: List[np.ndarray],
    labels: np.ndarray,
):
    path = os.path.join(
        "/assets", "activations", case_study, f"model_{model_id}", dataset
    )

    for layer_i, layer_at in enumerate(activations):
        folder = os.path.join(path, f"layer_{layer_i}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(os.path.join(folder, f"badge_{badge_id}.npy"), layer_at)

    labels_folder = os.path.join(path, "labels")
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    np.save(os.path.join(labels_folder, f"badge_{badge_id}.npy"), labels)


def persist(
    model: tf.keras.Sequential,
    case_study: str,
    model_id: int,
    train_set: Tuple[np.ndarray, np.ndarray],
    test_nominal: Tuple[np.ndarray, np.ndarray],
    test_corrupted: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Persist the activations of the model."""
    # Activations of all layers
    layer_ids = list(range(len(model.layers)))
    # Note: `include_last_layer = False` as already included in list above.
    transparent_model = BaseModel(
        model=model, activation_layers=layer_ids, include_last_layer=False
    )

    for ds, (x, y) in {
        "train": train_set,
        "test_nominal": test_nominal,
        "test_nominal_and_corrupted": test_corrupted,
    }.items():
        badge_stream = tf.data.Dataset.from_tensor_slices((x, y)).batch(BADGE_SIZE)
        print(
            f"Started collecting activations "
            f"for {badge_stream.cardinality()} {case_study}-{ds} badges."
        )
        for badge_id, (badge_x, badge_y) in enumerate(badge_stream.as_numpy_iterator()):
            activations = transparent_model.get_activations(badge_x)
            _persist_badge(
                case_study,
                model_id,
                ds,
                badge_id=badge_id,
                activations=activations,
                labels=badge_y,
            )
