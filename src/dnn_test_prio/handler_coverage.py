"""Logic to collect neuron coverages in an efficient way."""

import os
import secrets
import shutil
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tqdm as tqdm

from src.core.neuron_coverage import KMNC, NAC, NBC, SNAC, TKNC, CoverageMethod
from src.core.prioritizers import cam
from src.core.timer import Timer
from src.dnn_test_prio.aggregate_statistics import AggregateStatisticsCollector
from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.dnn_test_prio.handler_model import BaseModel


class CoverageWorker:
    """Effenciently handles Neuron Coverage instances."""

    def __init__(self, base_model: BaseModel, training_set: tf.data.Dataset):
        self.base_model = base_model
        self.metrics: Dict[str, CoverageMethod] = dict()
        self.setup_times: Dict[str, float] = dict()
        self.training_set = training_set
        # Using a random string to avoid collisions between different runs
        #   More cumbersome, we could have used case_study/model_id pairs,
        #   but the random string keeps the interface simpler
        self.temp_random = str(secrets.token_urlsafe(16))

        agg_stats = AggregateStatisticsCollector()
        pred_timer = Timer(start=True)
        for activations in tqdm.tqdm(
            base_model.walk_activations(training_set),
            "Walking over training set activations to calculate aggregate metrics",
        ):
            pred_timer.stop()
            agg_stats.track(activations)
            pred_timer.start()
        pred_timer.stop()

        print(
            "Done activation aggregate metrics. Creating coverage metric instances now..."
        )
        mins, maxs, std = agg_stats.get()

        nbc_debit = (
            agg_stats.min_timer.get()
            + agg_stats.max_timer.get()
            + pred_timer.get()
            + agg_stats.welford_timer.get()
        )
        self._add_metric(
            "NBC_0",
            lambda: NBC(mins=mins, maxs=maxs, stds=std, scaler=0),
            time_debit=nbc_debit,
        )
        self._add_metric(
            "NBC_0.5",
            lambda: NBC(mins=mins, maxs=maxs, stds=std, scaler=0.5),
            time_debit=nbc_debit,
        )
        self._add_metric(
            "NBC_1",
            lambda: NBC(mins=mins, maxs=maxs, stds=std, scaler=1),
            time_debit=nbc_debit,
        )

        snac_debit = (
            agg_stats.welford_timer.get() + agg_stats.max_timer.get() + pred_timer.get()
        )
        self._add_metric(
            "SNAC_0", lambda: SNAC(maxs=maxs, stds=std, scaler=0), time_debit=snac_debit
        )
        self._add_metric(
            "SNAC_0.5",
            lambda: SNAC(maxs=maxs, stds=std, scaler=0.5),
            time_debit=snac_debit,
        )
        self._add_metric(
            "SNAC_1", lambda: SNAC(maxs=maxs, stds=std, scaler=1), time_debit=snac_debit
        )

        self._add_metric("NAC_0", lambda: NAC(cov_threshold=0.0))
        self._add_metric("NAC_0.75", lambda: NAC(cov_threshold=0.75))

        self._add_metric("TKNC_1", lambda: TKNC(top_neurons=1))
        self._add_metric("TKNC_2", lambda: TKNC(top_neurons=2))
        self._add_metric("TKNC_3", lambda: TKNC(top_neurons=3))

        kmnc_debit = (
            agg_stats.min_timer.get() + agg_stats.max_timer.get() + pred_timer.get()
        )
        # KMNC_1000 and KMNC_10000 are used in the deepgini paper,
        #   but too large hyperparams to run within reasonable time.
        #   We use something smaller instead
        self._add_metric(
            "KMNC_2", lambda: KMNC(mins, maxs, sections=2), time_debit=kmnc_debit
        )

    def evaluate_all(
        self, test_dataset, test_dataset_id
    ) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray], Dict[str, List[int]]]:
        """Collect all the different neuron coverages for the passed datasets"""
        times, all_scores, cam_orders = dict(), dict(), dict()
        for metric_name, setup_time in self.setup_times.items():
            times[metric_name] = [setup_time, 0.0, 0.0]

        self._prepare_profiles(test_dataset, ds_id=test_dataset_id, times=times)
        for metric_id in self.metrics.keys():
            scores, profiles = self._load_prepared_profile(
                metric_id=metric_id, ds_id=test_dataset_id, delete=True
            )
            all_scores[metric_id] = scores
            # persist(dataset_id=test_dataset_id, data_type=f"{metric_id}_scores", model_id=model_id, data=scores)

            print(f"Calculating CAM for {metric_id}")
            timer = Timer()
            with timer:
                cam_orders[metric_id] = [
                    i for i in cam(scores=scores, profiles=profiles)
                ]
            times[metric_id].append(timer.get())

            self._cam_sanity_check(cam_orders[metric_id], scores)

            # persist(dataset_id=test_dataset_id, data_type=f"{metric_id}_cam_order", model_id=model_id,
            #         data=np.array(cam_order))
            del profiles  # This can be rather large and is not used anymore
        return times, all_scores, cam_orders

    def _get_temp_path(self, metric_id: str):
        return f"{OUTPUT_FOLDER}/.tmp/{self.temp_random}-prepared-profiles/{metric_id}"

    @staticmethod
    def _cam_sanity_check(cam_order, scores):
        assert (
            len(cam_order) == len(set(cam_order)) == scores.shape[0]
        ), "CAM order is not unique or not complete"

    def _add_metric(
        self,
        metric_id: str,
        metric_supplier: Callable[[], "CoverageMethod"],
        time_debit: float = 0.0,
    ):
        timer = Timer()
        with timer:
            self.metrics[metric_id] = metric_supplier()
        self.setup_times[metric_id] = time_debit + timer.get()

    def _timed_activation_walk(self, test_dataset):
        activations_generator = self.base_model.walk_activations(test_dataset)
        while True:
            try:
                timer = Timer()
                with timer:
                    activations = next(activations_generator)
                yield activations, timer.get()
            except StopIteration:
                return

    def _prepare_profiles(
        self, test_dataset: tf.data.Dataset, ds_id: str, times: Dict[str, List[float]]
    ):

        # Create empty folders for the profiles and scores
        for metric_id in self.metrics.keys():
            shutil.rmtree(f"{self._get_temp_path(metric_id)}", ignore_errors=True)
            os.makedirs(f"{self._get_temp_path(metric_id)}/{ds_id}-scores")
            os.makedirs(f"{self._get_temp_path(metric_id)}/{ds_id}-profiles")

        # Get Profiles (badge by batch to save memory)
        for b, (activations, pred_time) in enumerate(
            self._timed_activation_walk(test_dataset)
        ):
            for metric_id, metric in self.metrics.items():
                timer = Timer()
                with timer:
                    s, p = metric(activations)

                times[metric_id][1] += pred_time
                times[metric_id][2] += timer.get()
                np.save(f"{self._get_temp_path(metric_id)}/{ds_id}-scores/{b}.npy", s)
                np.save(f"{self._get_temp_path(metric_id)}/{ds_id}-profiles/{b}.npy", p)

    @staticmethod
    def _concatenate_arrays_in_folder(folder: str):
        """Concatenates all numpy arrays found in a specific folder"""
        arrays = [
            np.load(os.path.join(folder, f))
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
        return np.concatenate(arrays, axis=0)

    def _load_prepared_profile(self, metric_id: str, ds_id: str, delete: bool = True):
        folder = f"{self._get_temp_path(metric_id)}"
        scores = self._concatenate_arrays_in_folder(f"{folder}/{ds_id}-scores")
        profiles = self._concatenate_arrays_in_folder(f"{folder}/{ds_id}-profiles")
        if delete:
            shutil.rmtree(folder, ignore_errors=True)
        return scores, profiles
