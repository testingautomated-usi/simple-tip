import os
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.dnn_test_prio.handler_model import DROPOUT_SAMPLE_SIZE

N_FIRST_MODELS_CONSIDERED = 10


def load_times():
    """
    Load and times from the file system.
    :param case_study: The case study to load the times for.
    :return: A dictionary with the times.
    """
    #   folder and average them.
    times = dict()
    for root, dirs, files in os.walk(f"{OUTPUT_FOLDER}/times"):
        for file in files:
            file_san = (
                file.replace("softmax_entropy", "SE")
                .replace("pcs", "PCS")
                .replace("deep_gini", "DeepGini")
                .replace("softmax", "SM")
            )
            split = file_san.split("_")
            if len(split) == 5:
                case_study, dataset, model_id, metric, param = split
            else:
                # For things without param, such as uncertainty stuff
                case_study, dataset, model_id, metric = split
                param = ""

            if int(model_id) >= N_FIRST_MODELS_CONSIDERED:
                continue

            with open(os.path.join(root, file), "rb") as f:
                times[(case_study, dataset, model_id, metric, param)] = pickle.load(f)
    return times
