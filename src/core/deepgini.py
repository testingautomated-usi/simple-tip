"""Prioritization Scores which are based on Softmax DNN prediction.

They are implemented as uncertainty-wizard quantifiers.
Quantifiers already shipped with uncertainty-wizard (PCS, Softmax, ...)
are not re-implemented."""
from typing import List

import numpy as np
import uncertainty_wizard as uwiz


class DeepGini(uwiz.quantifiers.Quantifier):
    """DeepGini - Uncertainty (1 minus sum of squared softmax outputs)"""

    # docstr-coverage:inherited
    @classmethod
    def aliases(cls) -> List[str]:
        return ["custom::deep_gini"]

    # docstr-coverage:inherited
    @classmethod
    def takes_samples(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def is_confidence(cls) -> bool:
        return False

    # docstr-coverage:inherited
    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        predictions, _ = uwiz.quantifiers.MaxSoftmax.calculate(nn_outputs)
        gini = 1 - np.sum(nn_outputs * nn_outputs, axis=1)
        return predictions, gini

    # docstr-coverage:inherited
    @classmethod
    def problem_type(cls) -> uwiz.ProblemType:
        return uwiz.ProblemType.CLASSIFICATION
