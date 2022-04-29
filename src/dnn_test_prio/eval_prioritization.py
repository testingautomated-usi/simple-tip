"""The experimental logic related to test prioritization."""

import gc
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.dnn_test_prio.handler_coverage import CoverageWorker
from src.dnn_test_prio.handler_model import BaseModel
from src.dnn_test_prio.handler_surprise import SurpriseHandler

if not os.path.exists(f"{OUTPUT_FOLDER}/models/"):
    os.makedirs(f"{OUTPUT_FOLDER}/models/")
if not os.path.exists(f"{OUTPUT_FOLDER}/priorities/"):
    os.makedirs(f"{OUTPUT_FOLDER}/priorities/")


def _persist(
    case_study: str, dataset_id: str, data_type: str, model_id: int, data: np.ndarray
):
    """Stores the passed array on the file system"""
    return np.save(
        f"{OUTPUT_FOLDER}/priorities/{case_study}_{dataset_id}_{model_id}_{data_type}.npy",
        data,
    )


def _persist_times_multiple_metrics(
    case_study: str, dataset_id: str, model_id: int, data: Dict[str, List[float]]
):
    """Stores the passed array on the file system"""
    if not os.path.exists(f"{OUTPUT_FOLDER}/times/"):
        os.makedirs(f"{OUTPUT_FOLDER}/times/")
    # We don't write the dict as whole to make sure on partial re-run, nothing is lost
    for metric, times in data.items():
        with open(
            f"{OUTPUT_FOLDER}/times/{case_study}_{dataset_id}_{model_id}_{metric}", "wb"
        ) as f:
            pickle.dump(times, f)


def _persist_times(
    case_study: str, dataset_id: str, model_id: int, metric: str, data: List[float]
):
    with open(
        f"{OUTPUT_FOLDER}/times/{case_study}_{dataset_id}_{model_id}_{metric}", "wb"
    ) as f:
        pickle.dump(data, f)


def load(case_study: str, dataset_id: str, data_type: str, model_id: int) -> np.ndarray:
    """Loads the data from the file system"""
    return np.load(
        f"{OUTPUT_FOLDER}/priorities/{case_study}_{dataset_id}_{model_id}_{data_type}.npy"
    )


def evaluate(
    model_id: int,
    case_study: str,
    model: tf.keras.Model,
    training_dataset: tf.data.Dataset,
    nominal_test_dataset: tf.data.Dataset,
    nominal_test_labels: np.ndarray,
    ood_test_dataset: tf.data.Dataset,
    ood_test_labels: np.ndarray,
    nc_activation_layers: List[int],
    sa_activation_layers: List[int],
    dsa_badge_size: Optional[int] = None,
) -> None:
    """Run the experiments aiming to measure the TIPs test prioritization performance"""
    _eval_fault_predictors(
        case_study,
        model,
        model_id,
        nominal_test_dataset,
        nominal_test_labels,
        "nominal",
    )
    _eval_fault_predictors(
        case_study, model, model_id, ood_test_dataset, ood_test_labels, "ood"
    )

    _eval_neuron_coverage(
        case_study,
        model,
        model_id,
        nc_activation_layers,
        nominal_test_dataset,
        ood_test_dataset,
        training_dataset,
    )

    _eval_surprise(
        case_study,
        model,
        model_id,
        sa_activation_layers,
        nominal_test_dataset,
        ood_test_dataset,
        training_dataset,
        dsa_badge_size=dsa_badge_size,
    )

    del model
    tf.keras.backend.clear_session()
    gc.collect()


def _eval_surprise(
    case_study,
    model,
    model_id,
    layers,
    nominal_test_dataset,
    ood_test_dataset,
    training_dataset,
    dsa_badge_size: Optional[int] = None,
):
    sa_worker = SurpriseHandler(
        model=model, sa_layers=layers, training_dataset=training_dataset
    )
    results = sa_worker.evaluate_all(
        datasets={"nominal": nominal_test_dataset, "ood": ood_test_dataset},
        dsa_badge_size=dsa_badge_size,
    )

    for metric, values in results.items():
        for dataset, (sa, cam_order, times) in values.items():
            _persist_times(
                case_study=case_study,
                dataset_id=dataset,
                model_id=model_id,
                metric=metric,
                data=times,
            )
            _persist(
                case_study=case_study,
                dataset_id=dataset,
                data_type=f"{metric}_scores",
                model_id=model_id,
                data=sa,
            )
            _persist(
                case_study=case_study,
                dataset_id=dataset,
                data_type=f"{metric}_cam_order",
                model_id=model_id,
                data=cam_order,
            )


def _eval_neuron_coverage(
    case_study,
    model,
    model_id,
    layers,
    nominal_test_dataset,
    ood_test_dataset,
    training_dataset,
):
    nc_worker = CoverageWorker(
        base_model=BaseModel(model, activation_layers=layers),
        training_set=training_dataset,
    )
    for name, ds in {"nominal": nominal_test_dataset, "ood": ood_test_dataset}.items():
        times, scores, cam_orders = nc_worker.evaluate_all(ds, name)
        _persist_times_multiple_metrics(
            case_study=case_study, dataset_id=name, model_id=model_id, data=times
        )
        for metric_id, score in scores.items():
            _persist(
                case_study=case_study,
                dataset_id=name,
                data_type=f"{metric_id}_scores",
                model_id=model_id,
                data=score,
            )
        for metric_id, order in cam_orders.items():
            _persist(
                case_study=case_study,
                dataset_id=name,
                data_type=f"{metric_id}_cam_order",
                model_id=model_id,
                data=np.array(order),
            )


def _eval_fault_predictors(case_study, model, model_id, ds, labels, ds_type):
    base_model = BaseModel(model, activation_layers=None)
    pred, uncertainties, times = base_model.get_pred_and_uncertainty(ds)
    is_misclassified = pred != labels

    _persist(
        case_study=case_study,
        dataset_id=ds_type,
        data_type="is_misclassified",
        model_id=model_id,
        data=is_misclassified,
    )
    _persist_times_multiple_metrics(
        case_study=case_study, dataset_id=ds_type, model_id=model_id, data=times
    )
    for unc_id, unc in uncertainties.items():
        _persist(
            case_study=case_study,
            dataset_id=ds_type,
            data_type=f"uncertainty_{unc_id}",
            model_id=model_id,
            data=unc,
        )
