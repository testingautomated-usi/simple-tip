"""The experimental logic related to active learning."""

import gc
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.dnn_test_prio.handler_coverage import CoverageWorker
from src.dnn_test_prio.handler_model import BaseModel
from src.dnn_test_prio.handler_surprise import SurpriseHandler

RANDOM_SPLIT = "random"

SplitDataset = Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]
"""Alias. Use as: { ("ood"/"nominal", "observed"/"future") : (x, y) } """

SplitEvaluation = Dict[Tuple[str, str], float]
"""Alias. Use as: { ("ood"/"nominal", "observed"/"future") : accuracy } """

MetricSelection = Dict[Tuple[str, str], List[int]]
"""Alias. Use as: { (metric, "ood"/"nominal") : selected_indexes } """

NOM = "nominal"
OOD = "ood"

OBS = "observed"
FUT = "future"


def evaluate(
    model_id: int,
    case_study: str,
    model: tf.keras.Model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    nominal_test_x: np.ndarray,
    nominal_test_labels: np.ndarray,
    ood_test_x: np.ndarray,
    ood_test_labels: np.ndarray,
    nc_activation_layers: List[int],
    sa_activation_layers: List[int],
    training_process: Callable[[np.ndarray, np.ndarray], tf.keras.Model],
    observed_share: float,
    num_selected: int,
    num_classes: Optional[int],
    dsa_badge_size: Optional[int] = None,
) -> None:
    """Evaluate the active learning capabilities"""

    # Shuffle and split the test sets into observed and future
    active_datasets = _shuffle_and_split_datasets(
        model_id,
        nominal_test_x,
        nominal_test_labels,
        ood_test_x,
        ood_test_labels,
        observed_share=observed_share,
    )

    # Measure accuracies on the observed and future test sets
    #  with the original model
    original_model_eval = _evaluate(model, active_datasets, num_classes)

    # Prioritize the observed test sets
    train_x_tfds = tf.data.Dataset.from_tensor_slices(train_x).batch(128)
    selections: MetricSelection = dict()
    selections.update(_get_fp_selection(model, active_datasets, num_selected))
    selections.update(
        _get_nc_selection(
            model, train_x_tfds, active_datasets, nc_activation_layers, num_selected
        )
    )
    selections.update(
        _get_sa_selection(
            model,
            train_x_tfds,
            active_datasets,
            sa_activation_layers,
            num_selected,
            dsa_badge_size,
        )
    )
    selections.update(_get_random_section(active_datasets, num_selected))

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    # Run some sanity checks on the selections,
    #  to make sure that problems are detected, before starting the retraining
    _selection_sanity_checks(num_selected, selections)

    # Retrain the model with the prioritized test sets
    #  and measure accuracies on the observed and future test sets, for both categories
    active_accuracies = dict()
    for (metric, ood_or_nom), selected_indexes in selections.items():
        x = active_datasets[ood_or_nom, OBS][0][selected_indexes]
        y = active_datasets[ood_or_nom, OBS][1][selected_indexes]
        new_model = _retrain(num_classes, training_process, train_x, train_y, x, y)
        # Note: We evaluate every model on all four splits
        #   (ood-observed, ood-future, nominal-observed, nominal-future)
        #   even though we only need one or two of them.
        #   But the other ones are now cheap to get now, and might become
        #   interesting artifacts in the future.
        #   Rerunning everything very expensive, hence we better keep it now.
        improved_model_results = _evaluate(new_model, active_datasets, num_classes)
        active_accuracies[(metric, ood_or_nom)] = improved_model_results

        del new_model
        tf.keras.backend.clear_session()
        gc.collect()

    _save_results_on_file(
        case_study=case_study,
        model_id=model_id,
        metric="original",
        ood_or_nom="na",
        eval_res=original_model_eval,
    )
    for (metric, ood_or_nom), eval_res in active_accuracies.items():
        _save_results_on_file(
            case_study=case_study,
            model_id=model_id,
            metric=metric,
            ood_or_nom=ood_or_nom,
            eval_res=eval_res,
        )


def _save_results_on_file(
    case_study: str,
    model_id: int,
    metric: str,
    ood_or_nom: str,
    eval_res: SplitEvaluation,
) -> None:
    """Save the results to the file system"""
    path = (
        f"{OUTPUT_FOLDER}/active_learning/"
        f"{case_study}_{model_id}_{metric}_{ood_or_nom}.pickle"
    )
    with open(path, "wb") as f:
        pickle.dump(eval_res, f)


def _selection_sanity_checks(num_selected, selections):
    for (metric, ood_or_nom), selected_idx in selections.items():
        assert len(selected_idx) == num_selected, (
            f"The number of selected indexes for {metric}, {ood_or_nom} is not correct."
            f"Should be {num_selected}, but was {len(selected_idx)}"
        )
        assert (
            len(set(selected_idx)) == num_selected
        ), f"The number of selected indexes for {metric}, {ood_or_nom} is not unique."


def _retrain(num_classes, training_process, train_x, train_y, new_x, new_y):
    """Retrain the model including the selected the new data"""
    # Merge the two sets
    x = np.concatenate((train_x, new_x))

    # Make sure we do not flatten away non-empty dimensions,
    #   which would indicate a bug in the code
    assert train_y.shape[0] == np.prod(train_y.shape)
    assert new_y.shape[0] == np.prod(new_y.shape)
    y = np.concatenate((train_y.flatten(), new_y.flatten()))
    # Shuffle the new array
    shuffled_idx = np.random.permutation(len(x))
    x = x[shuffled_idx]
    y = y[shuffled_idx]
    # One-Hot encode the labels
    if num_classes is not None:
        y = tf.keras.utils.to_categorical(y, num_classes)
    # Train the model
    new_model = training_process(x, y)
    return new_model


def _get_random_section(dataset: SplitDataset, num_selected: int) -> MetricSelection:
    """A random selection of observed samples, serves as benchmark."""
    res: MetricSelection = dict()
    for (ood_or_nom, observed_or_future), (x, y) in dataset.items():
        if observed_or_future == OBS:
            # The arrays are already shuffled
            res["random", ood_or_nom] = [x for x in range(num_selected)]
    return res


def _get_fp_selection(
    model: tf.keras.Model, datasets: SplitDataset, num_selected: int
) -> MetricSelection:
    """Selection of observed samples based on fault predictors"""
    res: MetricSelection = dict()
    base_model = BaseModel(model, activation_layers=None)

    for (ood_or_nom, observed_or_future), (x, y) in datasets.items():
        if observed_or_future == OBS:
            # The arrays are already shuffled
            ds = tf.data.Dataset.from_tensor_slices(x).batch(128)
            _, uncertainties, _ = base_model.get_pred_and_uncertainty(ds)
            for metric, uncertainty in uncertainties.items():
                # Select the samples with the highest uncertainty
                res[metric, ood_or_nom] = np.argsort(uncertainty)[-num_selected:]

    return res


def _get_nc_selection(
    model: tf.keras.Model,
    training_dataset: tf.data.Dataset,
    datasets: SplitDataset,
    nc_activation_layers: List[int],
    num_selected: int,
) -> MetricSelection:
    """Selection of observed samples based on neuron coverage"""
    res: MetricSelection = dict()

    nc_worker = CoverageWorker(
        base_model=BaseModel(model, activation_layers=nc_activation_layers),
        training_set=training_dataset,
    )
    for (ood_or_nom, observed_or_future), (x, y) in datasets.items():
        if observed_or_future == OBS:
            # The arrays are already shuffled
            ds = tf.data.Dataset.from_tensor_slices(x).batch(128)
            _, all_scores, cam_orders = nc_worker.evaluate_all(ds, num_selected)
            for metric, scores in all_scores.items():
                # Select the samples with the highest score
                res[metric, ood_or_nom] = np.argsort(scores)[-num_selected:]

            for metric, cam_order in cam_orders.items():
                # Select the first num_selected samples in the CAM order
                res[f"{metric}-cam", ood_or_nom] = cam_order[:num_selected]

    return res


def _get_sa_selection(
    model: tf.keras.Model,
    training_dataset: tf.data.Dataset,
    datasets: SplitDataset,
    sa_activation_layers: List[int],
    num_selected: int,
    dsa_badge_size: Optional[int] = None,
) -> MetricSelection:
    """Selection of observed samples based on surprise adequacy and coverage"""
    res: MetricSelection = dict()
    sa_worker = SurpriseHandler(
        model=model, sa_layers=sa_activation_layers, training_dataset=training_dataset
    )
    results = sa_worker.evaluate_all(
        datasets={
            NOM: tf.data.Dataset.from_tensor_slices(datasets[NOM, OBS][0]).batch(128),
            OOD: tf.data.Dataset.from_tensor_slices(datasets[OOD, OBS][0]).batch(128),
        },
        dsa_badge_size=dsa_badge_size,
    )

    for metric, values in results.items():
        for nom_or_ood, (sa, cam_order, _) in values.items():
            # Select the samples with the highest surprise adequacy
            res[metric, nom_or_ood] = np.argsort(sa)[-num_selected:]

            # Select the first num_selected samples in the CAM order
            res[f"{metric}-cam", nom_or_ood] = cam_order[:num_selected]
    return res


def _shuffle_and_split_datasets(
    model_id: int,
    nominal_x: np.ndarray,
    nominal_y: np.ndarray,
    ood_x: np.ndarray,
    ood_y: np.ndarray,
    observed_share: float,
) -> SplitDataset:
    """Shuffle and split the test sets into observed and future splits"""
    res: SplitDataset = dict()

    fut_x, obs_x, fut_y, obs_y = train_test_split(
        nominal_x, nominal_y, test_size=observed_share, random_state=model_id
    )
    res[NOM, OBS] = (obs_x, obs_y)
    res[NOM, FUT] = (fut_x, fut_y)

    fut_x, obs_x, fut_y, obs_y = train_test_split(
        ood_x, ood_y, test_size=observed_share, random_state=model_id
    )
    res[OOD, OBS] = (obs_x, obs_y)
    res[OOD, FUT] = (fut_x, fut_y)

    return res


def _evaluate(
    model: tf.keras.Model, datasets: SplitDataset, num_classes: int
) -> SplitEvaluation:
    """Evaluate the models accuracy on all four dataset splits."""
    res: SplitEvaluation = dict()
    for (ood_or_nom, observed_or_future), (x, y) in datasets.items():
        if num_classes is not None:
            y = tf.keras.utils.to_categorical(y, num_classes)
        acc = model.evaluate(x, y)[1]
        assert 0 <= acc <= 1, (
            "The models metric is not accuracy, change your "
            "training_process callable."
        )
        res[ood_or_nom, observed_or_future] = acc
    return res
