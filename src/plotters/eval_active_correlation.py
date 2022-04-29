"""Evaluates p-values and effect sizes for the active learning performances."""

import os
import pickle
import warnings
from typing import Dict, List

import pandas as pd

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.plotters import utils
from src.plotters.correlation_plot import WilcoxonCorrelationPlot
from src.plotters.eval_active_learning_table import load_arrays_active_learning
from src.plotters.utils import identify_incomplete_values, named_tuples

GET_VALUES = True
"""`False` indicates the use of pre-loaded values for speedup."""


def _load(case_study: str, dataset: str):
    res: Dict[str, Dict[int, float]] = {
        approach: dict() for approach in utils.APPROACHES
    }
    res["original"] = dict()
    res["random"] = dict()
    loaded = load_arrays_active_learning(case_study, dataset, by_id=True)
    for i in range(100):
        for approach, l in loaded.items():
            if i in loaded[approach]:
                # We do not check significance for the observed or cross-dataset
                #  splits, but the script can be easily adapted to do so
                #  by changing the following line
                split_key = (dataset, "future")
                res[approach][i] = loaded[approach][i][split_key]

    # print(res)
    return res


def _print_missing_values(cs, ds, values):
    missing_values = identify_incomplete_values(values, has_dropout=cs != "cifar10")
    if len(missing_values) > 0:
        print(f"Missing values {cs} - {ds}: {missing_values}")


def run():
    """Runs this module. Called by this module main method, or reproduction CLI."""
    temp_data_store = OUTPUT_FOLDER + ".tmp/active_for_correlation_plot"
    if GET_VALUES or not os.path.exists(temp_data_store):
        vals: List[Dict[str, Dict[str, float]]] = []
        for cs in ["mnist", "fmnist", "cifar10", "imdb"]:
            for ds in ["nominal", "ood"]:
                values = _load(cs, ds)
                _print_missing_values(cs, ds, values)

                approaches = utils.APPROACHES.copy()
                approaches.extend(["original", "random"])
                named = named_tuples(cs, values, None, approaches=approaches)
                vals.append(named)

        # flatten the vals list
        all_by_approach: Dict[str, Dict[str, float]] = dict()
        for named in vals:
            for approach, data in named.items():
                if approach not in all_by_approach:
                    all_by_approach[approach] = dict()
                all_by_approach[approach].update(data)

        # Save (pickle) the all_by_approach dict to file
        with open(temp_data_store, "wb") as f:
            pickle.dump(all_by_approach, f)

        print("Done loading values")

    else:
        warnings.warn(
            "Attention: Loading cached active learning values from file. "
            "You should not see this when reproducing results."
        )
        with open(temp_data_store, "rb") as f:
            all_by_approach = pickle.load(f)
    print("Plotting correlation heatmap...")
    pseudo_plot = WilcoxonCorrelationPlot(
        approaches=utils.CORRELATION_PLOT_APPROACHES, num_tested_approaches=39
    )
    for approach, data in all_by_approach.items():
        for measurement, value in data.items():
            pseudo_plot.add_measurement(approach, measurement, value)
    pseudo_plot.plot_heatmap("active", "all", "both")
    print("Calculating and storing full result csv...")
    pseudo_plot = WilcoxonCorrelationPlot(
        approaches=utils.APPROACHES, num_tested_approaches=39
    )
    for approach, data in all_by_approach.items():
        for measurement, value in data.items():
            pseudo_plot.add_measurement(approach, measurement, value)
    p_and_eff = pseudo_plot.calc_values()
    human_readable_approaches = utils.human_approach_names(utils.APPROACHES)
    p_pd = pd.DataFrame(
        data=p_and_eff["p"],
        index=human_readable_approaches,
        columns=human_readable_approaches,
    )
    p_pd.replace(10000, "", inplace=True)
    p_pd.to_csv(OUTPUT_FOLDER + "/results/active_correlation_p.csv")
    e_pd = pd.DataFrame(
        data=p_and_eff["e"],
        index=human_readable_approaches,
        columns=human_readable_approaches,
    )
    p_pd.replace(10000, "", inplace=True)
    e_pd.to_csv(OUTPUT_FOLDER + "/results/active_correlation_eff.csv")
    print("Done!")


if __name__ == "__main__":
    run()
