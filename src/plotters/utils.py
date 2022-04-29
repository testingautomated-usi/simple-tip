"""Utilities to be used by, but not specific to, one or multiple result evaluators."""

import logging
import os
import pickle
from re import Pattern
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.dnn_test_prio.case_study import OUTPUT_FOLDER

NUM_RUNS = 100

VERTI_DEF = (
    "\\newcommand{\\verti}[1]{\\begin{tabular}{@{}c@{}}"
    "\\rotatebox[origin=c]{90}{\centering #1}\end{tabular}}"
)

# All approaches tested in our experiments
APPROACHES = [
    "NAC_0.75-cam",
    "NAC_0.75",
    "NAC_0-cam",
    "NAC_0",
    "NBC_0.5-cam",
    "NBC_0.5",
    "NBC_0-cam",
    "NBC_0",
    "NBC_1-cam",
    "NBC_1",
    "SNAC_0.5-cam",
    "SNAC_0.5",
    "SNAC_0-cam",
    "SNAC_0",
    "SNAC_1-cam",
    "SNAC_1",
    "TKNC_1-cam",
    "TKNC_1",
    "TKNC_2-cam",
    "TKNC_2",
    "TKNC_3-cam",
    "TKNC_3",
    "KMNC_2-cam",
    "KMNC_2",
    "dsa-cam",
    "dsa",
    "pc-lsa-cam",
    "pc-lsa",
    "pc-mdsa-cam",
    "pc-mdsa",
    "pc-mlsa-cam",
    "pc-mlsa",
    "pc-mmdsa-cam",
    "pc-mmdsa",
    "deep_gini",
    "softmax",
    "pcs",
    "softmax_entropy",
    "VR",
]

# Paper-Table Approaches
PAPER_APPROACHES = [
    "NAC_0.75-cam",
    "NAC_0.75",
    "NBC_0-cam",
    "NBC_0",
    "SNAC_0-cam",
    "SNAC_0",
    "TKNC_1-cam",
    "KMNC_2",
    "dsa",
    "pc-lsa",
    "pc-mdsa",
    "pc-mlsa",
    "pc-mmdsa",
    "deep_gini",
    "softmax",
    "pcs",
    "softmax_entropy",
    "VR",
]

# Paper-Correlation Plot Approaches
CORRELATION_PLOT_APPROACHES = [
    # 3 best-on-average NC Metrics
    "SNAC_0",
    "SNAC_0-cam",
    "NBC_0-cam",
    # 3 best-on-average non-cam surprise adequacy Metrics
    "dsa",
    "pc-mdsa",
    "pc-mlsa",
    # 3 best-on-average point-prediction uncertainty metrics
    "deep_gini",
    "softmax",
    "softmax_entropy",
]


def human_appraoch_name(approach: str) -> str:
    """Converts the internally used approach names to the ones used in the paper."""
    if approach == "softmax_entropy":
        return "Entropy"
    elif approach == "VR":
        return "MC-Dropout"
    elif approach == "softmax":
        return "Vanilla SM"
    elif approach == "deep_gini":
        return "DeepGini"
    elif approach in ["uncertainty", "surprise", "neuron coverage", "baseline"]:
        return approach
    else:
        return approach.replace("_", "-").upper()


def human_approach_names(approaches: List[str]) -> List[str]:
    """Converts the internally used approach names to the ones used in the paper."""
    return [human_appraoch_name(approach) for approach in approaches]


def approach_name(approach: str, param: str = "", cam: bool = False) -> str:
    """Creates approach names which also show TIP parameters (such as k and possibly CAM)."""
    res = approach
    if param:
        res += f"_{param}"
    if cam:
        res += "-cam"
    return res


def _row(approach: str) -> Tuple[str, str]:
    return category(approach), approach


def category(approach: str) -> str:
    """Returns the category of the approach."""
    if approach in ["deep_gini", "softmax", "pcs", "softmax_entropy", "VR"]:
        return "uncertainty"
    if approach in [
        "dsa-cam",
        "dsa",
        "pc-lsa-cam",
        "pc-lsa",
        "pc-mdsa-cam",
        "pc-mdsa",
        "pc-mlsa-cam",
        "pc-mlsa",
        "pc-mmdsa-cam",
        "pc-mmdsa",
    ]:
        return "surprise"
    if approach in ["original", "random"]:
        return "baseline"
    if any([approach.startswith(nc) for nc in ["NAC", "NBC", "SNAC", "TKNC", "KMNC"]]):
        return "neuron coverage"


def vertical_categories(latex: str) -> str:
    """Changes orientation of the cells representing TIP categories in a latex string."""
    latex = VERTI_DEF + latex
    for category in ["uncertainty", "surprise", "baseline", "neuron coverage"]:
        latex = latex.replace(category, "\\verti{" + category + "}", 1)
    return latex


def load_all_for_regex(research_question: str, regex: Pattern) -> Tuple[List, List]:
    """Loads all results for a given research question and a given regex."""
    file_contents = []
    matches = []
    for root, dirs, files in os.walk(f"{OUTPUT_FOLDER}/{research_question}"):
        for file in files:
            if regex.match(file, pos=0):
                matches.append(file)
                if file.endswith(".npy"):
                    file_contents.append(
                        np.load(f"{OUTPUT_FOLDER}/{research_question}/{file}")
                    )
                else:
                    # Unpickle file
                    with open(f"{OUTPUT_FOLDER}/{research_question}/{file}", "rb") as f:
                        file_contents.append(pickle.load(f))
    return file_contents, matches


def identify_incomplete_values(
    data: Dict[str, Dict[int, float]], has_dropout: bool
) -> Set[int]:
    """Identifies the indices of the data which are incomplete. Used to sanity check the artifacts."""
    # Tuple[Dict[str, List[float]], Set[int]]:
    missing_or_incomplete_runs = set()
    for approach, runs in data.items():
        for i in range(NUM_RUNS):
            try:
                runs[i]
            except KeyError:
                if approach != "VR" or has_dropout:
                    missing_or_incomplete_runs.add(i)

    return missing_or_incomplete_runs


def named_tuples(
    cs_data_id: str,
    data: Dict[str, Dict[int, float]],
    collection: Optional[Dict[str, float]],
    approaches: List[str],
) -> Dict[str, Dict[str, float]]:
    """Creates a clearly identifiable data structer of intermediate results."""
    if collection is None:
        collection = dict()
        for approach in approaches:
            collection[approach] = dict()
    else:
        for approach in approaches:
            assert approach in collection.keys()

    for approach, runs in data.items():
        for run_id, value in runs.items():
            unique_id = f"{cs_data_id}_{run_id}"
            if unique_id in collection[approach]:
                logging.warning(f"{cs_data_id}: Run {unique_id} already in collection")
            else:
                collection[approach][unique_id] = value

    return collection
