"""Responsible for training of, and predictions on DNN models"""
import logging

from src.core.deepgini import DeepGini
from src.core.timer import Timer

DROPOUT_SAMPLE_SIZE = 200

"""Collects the predictions and inner activations of a passed model"""
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

try:
    uwiz.quantifiers.QuantifierRegistry.register(DeepGini())
except ValueError as e:
    if "already registered" in str(e):
        pass


class TimedPCS(uwiz.quantifiers.PredictionConfidenceScore):
    """Wrapper for quantifiers which time their calculations"""

    def __init__(self):
        super().__init__()
        self.timer = Timer()

    # docstr-coverage: inherited
    def calculate(self, nn_outputs: np.ndarray):
        with self.timer:
            return super().calculate(nn_outputs)


class TimedMS(uwiz.quantifiers.MaxSoftmax):
    """Wrapper for quantifiers which time their calculations"""

    def __init__(self):
        super().__init__()
        self.timer = Timer()

    # docstr-coverage: inherited
    def calculate(self, nn_outputs: np.ndarray):
        with self.timer:
            return super().calculate(nn_outputs)


class TimedSE(uwiz.quantifiers.SoftmaxEntropy):
    """Wrapper for quantifiers which time their calculations"""

    def __init__(self):
        super().__init__()
        self.timer = Timer()

    # docstr-coverage: inherited
    def calculate(self, nn_outputs: np.ndarray):
        with self.timer:
            return super().calculate(nn_outputs)


class TimedGini(DeepGini):
    """Wrapper for quantifiers which time their calculations"""

    def __init__(self):
        super().__init__()
        self.timer = Timer()

    # docstr-coverage: inherited
    def calculate(self, nn_outputs: np.ndarray):
        with self.timer:
            return super().calculate(nn_outputs)


class TimedVR(uwiz.quantifiers.VariationRatio):
    """Wrapper for quantifiers which time their calculations"""

    def __init__(self):
        super().__init__()
        self.timer = Timer()

    # docstr-coverage: inherited
    def calculate(self, nn_outputs: np.ndarray):
        with self.timer:
            return super().calculate(nn_outputs)


class BaseModel:
    """Model centric class providing various utilities, such as activation collection."""

    def __init__(
        self,
        model: tf.keras.Sequential,
        activation_layers: Optional[List[int]],
        include_last_layer: bool = False,
    ):
        self.sequential_model: tf.keras.model.Sequential = model
        self.activation_layers: List[int] = activation_layers
        self.transparent_model: Optional[tf.keras.model.Functional] = None
        self.include_last_layer: bool = include_last_layer

    def get_pred_and_uncertainty(
        self, x: tf.data.Dataset
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, List[float]]]:
        """Point predictions and uncertainties as determined by Point-Pred Quantifiers."""
        det_quantifiers = [TimedMS(), TimedPCS(), TimedSE(), TimedGini()]

        try:
            stochastic_model = uwiz.models.stochastic_from_keras(self.sequential_model)
            has_stochastic_layers = True
        except ValueError as e:
            if "no stochastic layers" in str(e):
                logging.warning(
                    "No stochastic layers found in model. Skipping stochastic quantifiers."
                )
                stochastic_model = uwiz.models.stochastic_from_keras(
                    self.sequential_model, expect_determinism=True
                )
                has_stochastic_layers = False
            else:
                raise e
        # Batching will be done by uncertainty wizard
        pred_timer = Timer()
        logging.info("Collecting Point Pred quantifications")

        try:
            batch_size = self.sequential_model.custom_badge_size
        except AttributeError:
            # No custom badge size specified, so use default
            batch_size = 32

        with pred_timer:
            res = stochastic_model.predict_quantified(
                x.unbatch(),
                quantifier=det_quantifiers,
                as_confidence=False,
                batch_size=batch_size,
                verbose=1,
            )
        pred_time = pred_timer.get() - sum(q.timer.get() for q in det_quantifiers)
        uncertainties = dict()
        times = dict()
        for i, r in enumerate(res):
            name = det_quantifiers[i].aliases()[0].replace("custom::", "")
            uncertainties[name] = r[1]
            times[name] = [0, pred_time, det_quantifiers[i].timer.get(), 0]

        if has_stochastic_layers:
            logging.info("Collecting MC-Dropout samples")
            sampling_timer = Timer()
            timed_vr = TimedVR()
            with sampling_timer:
                # Batching will be done by uncertainty wizard
                _, vr_unc = stochastic_model.predict_quantified(
                    x.unbatch(),
                    quantifier=timed_vr,
                    sample_size=DROPOUT_SAMPLE_SIZE,
                    as_confidence=False,
                    verbose=1,
                    batch_size=batch_size,
                )
            logging.info("Done quantifying")
            uncertainties["VR"] = vr_unc
            # [setup_time, pred_time, online_time, cam_time]
            sampling_time = sampling_timer.get() - timed_vr.timer.get()
            times["VR"] = [0, sampling_time, timed_vr.timer.get(), 0]

        # We return the point prediction and the (point pred and stochastic) uncertainties
        pred = res[0][0]
        if len(pred.shape) != 1:
            assert pred.shape[0] == np.prod(pred.shape)
            pred = pred.flatten()
        return pred, uncertainties, times

    def walk_activations(
        self, x: tf.data.Dataset
    ) -> Generator[List[np.ndarray], None, None]:
        """Walks over the activations of a given (potentially large) dataset"""
        for badge in x.as_numpy_iterator():
            yield self.get_activations(badge)

    def get_activations(
        self, x: Union[np.ndarray, tf.data.Dataset]
    ) -> List[np.ndarray]:
        """Makes a deterministic prediction, returning all layer activations.

        For a more scalable method, call walk_activations"""
        if self.transparent_model is None:
            self._init_transparent_model()

        return self.transparent_model.predict(x)

    def _init_transparent_model(self):
        """Creates a transparent model from the sequential model."""
        if self.activation_layers is None:
            raise ValueError("No activation layers specified")

        inp = self.sequential_model.input
        outputs = [
            layer.output
            for i, layer in enumerate(self.sequential_model.layers)
            if i in self.activation_layers
        ]
        if self.include_last_layer:
            outputs.append(self.sequential_model.output)
        self.transparent_model = tf.keras.Model(inputs=inp, outputs=outputs)
