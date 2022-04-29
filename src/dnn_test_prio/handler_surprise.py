"""Logic to collect surprise adequacies and surprise coverages in an efficient way."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from src.core.prioritizers import cam
from src.core.surprise import DSA, LSA, MDSA, MLSA, MultiModalSA, SurpriseCoverageMapper
from src.core.timer import Timer
from src.dnn_test_prio.handler_model import BaseModel

NUM_SC_BUCKETS = 1000

import tensorflow as tf


class SurpriseHandler:
    """Effenciently handles Surprise Adequacy instances."""

    TESTED_SA = {
        # Plain Distance-Based Surprise Adequacy
        "dsa": lambda x, y: DSA(x, y, subsampling=0.3),
        # Per-Class Likelihood Surprise Adequacy
        "pc-lsa": lambda x, y: MultiModalSA.build_by_class(x, y, lambda x, y: LSA(x)),
        # Per-Class  Mahalanobis Distance based Surprise Adequacy
        "pc-mdsa": lambda x, y: MultiModalSA.build_by_class(x, y, lambda x, y: MDSA(x)),
        # # Per-Class  Multimodal Likelihood Surprise Adequacy
        "pc-mlsa": lambda x, y: MultiModalSA.build_by_class(
            x, y, lambda x, y: MLSA(x, num_components=3)
        ),
        # # Per-Class  Multimodal Mahalanobis Distance based Surprise Adequacy
        "pc-mmdsa": lambda x, y: MultiModalSA.build_with_kmeans(
            x, y, lambda x, y: MDSA(x), potential_k=range(2, 6), subsampling=0.3
        ),
    }

    def __init__(
        self,
        model: tf.keras.Sequential,
        sa_layers: List[int],
        training_dataset: Union[np.ndarray, tf.data.Dataset],
    ):
        self.sa_layers = sa_layers.copy()
        self.base_model = BaseModel(model, self.sa_layers, include_last_layer=True)
        self.train_at_timer = Timer()
        with self.train_at_timer:
            self.train_ats, self.train_pred = self._acti_and_pred(training_dataset)

    def _acti_and_pred(
        self, dataset: Union[np.ndarray, tf.data.Dataset]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Collects activations and predicions in a single NN forward pass"""
        outputs = self.base_model.get_activations(dataset)
        assert len(outputs) == len(self.sa_layers) + 1
        return (outputs[:-1], np.argmax(outputs[-1], axis=1))

    def evaluate_all(
        self,
        datasets: Dict[str, Union[np.ndarray, tf.data.Dataset]],
        dsa_badge_size: Optional[int] = None,
    ):
        """Collect all the different surprise adequacies for the passed datasets"""
        res = dict()
        # ats, predictions, times
        test_apt = dict()

        # Map test dataset to ats and preds
        logging.info(f"Collecting SA ATs")
        for ds_name, dataset in datasets.items():
            test_pred_timer = Timer()
            with test_pred_timer:
                test_ats, test_pred = self._acti_and_pred(dataset)
            test_apt[ds_name] = (test_ats, test_pred, test_pred_timer.get())

        # Calc SAs
        for sa_name, sa_func in tqdm(self.TESTED_SA.items(), desc="Calculating SAs"):
            res[sa_name] = dict()
            setup_timer = Timer()
            with setup_timer:
                logging.info(f"Creating {sa_name} instance")
                sa = sa_func(self.train_ats, self.train_pred)
                if isinstance(sa, DSA) and dsa_badge_size is not None:
                    sa.badge_size = dsa_badge_size
            setup_time = self.train_at_timer.get() + setup_timer.get()

            for ds_name, (test_ats, test_pred, test_pred_timer) in test_apt.items():
                sa_timer = Timer()
                with sa_timer:
                    logging.info(f"Calculating {sa_name} for {ds_name}")
                    sa_pred = sa(test_ats, test_pred)

                times = [setup_time, test_apt[ds_name][2], sa_timer.get()]
                res[sa_name][ds_name] = (sa_pred, times)

        # Calc CAMs
        # We're interating over the keys of the inputs,
        #   instead of the items of res, to avoid modifying the dict which
        #   is being iterated over.
        for sa_name in self.TESTED_SA.keys():
            for ds_name in datasets.keys():
                sa_pred, times = res[sa_name][ds_name]
                cam_timer = Timer()
                with cam_timer:
                    # We use the max of all observed values to dynamically select the
                    #   buckets upper bound.
                    coverage_mapper = SurpriseCoverageMapper(
                        NUM_SC_BUCKETS, np.max(sa_pred)
                    )
                    coverage_profiles = coverage_mapper.get_coverage_profile(sa_pred)
                    cam_order = [i for i in cam(sa_pred, coverage_profiles)]
                cam_order = np.array(cam_order)
                times.append(cam_timer.get())
                res[sa_name][ds_name] = (sa_pred, cam_order, times)

        return res
