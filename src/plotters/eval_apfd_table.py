"""Evaluates APFD performances (table 1)."""
import os
import warnings
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import pandas
import pandas as pd

from src.core.apfd import apfd_from_order
from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.plotters import times_collector
from src.plotters.utils import (
    APPROACHES,
    PAPER_APPROACHES,
    _row,
    approach_name,
    human_appraoch_name,
    vertical_categories,
)

TIME_COL = "time"

FIRST_K_MODELS_CONSIDERED = 100


def _sanitize_metric_name(metric: str) -> str:
    switcher = {
        "SE": "Entropy",
        "SM": "Max SM",
        "VR": "MC-Dropout",
    }
    try:
        return switcher[metric]
    except KeyError:
        return metric


SETTING = Tuple[str, str, str, str, str]


def load_apfd_values(case_study: str, ds_name: str) -> Dict[str, Dict[int, float]]:
    """
    Load and times from the file system.
    :param case_study: The case study to load the times for.
    :return: A dictionary with the times.
    """

    #   folder and average them.
    misclassifications = dict()
    orders = dict()

    for root, dirs, files in os.walk(f"{OUTPUT_FOLDER}/priorities"):
        for file in files:
            if not file.endswith(".npy"):
                continue
            if not file.startswith(f"{case_study}_{ds_name}"):
                continue
            arr = np.load(os.path.join(root, file))
            if file.endswith("is_misclassified.npy"):
                case_study, dataset, model_id, _, _ = file.split("_")
                if int(model_id) < FIRST_K_MODELS_CONSIDERED:
                    misclassifications[model_id] = arr
            elif file.endswith("cam_order.npy"):
                if "dsa" in file or "lsa" in file:
                    _, _, model_id, metric, _, _ = file.split("_")
                    metric = approach_name(metric, cam=True)
                else:
                    _, _, model_id, metric, param, _, _ = file.split("_")
                    metric = approach_name(metric, param=param, cam=True)
                orders[(metric, model_id)] = arr
            else:
                # scores
                param = ""
                if "uncertainty" in file:
                    file = file.replace(".npy", "").replace(
                        f"{case_study}_{ds_name}_", ""
                    )
                    model_id, metric = file.split("_uncertainty_")
                elif "dsa" in file or "lsa" in file:
                    _, _, model_id, metric, _ = file.split("_")
                else:
                    _, _, model_id, metric, param, _ = file.split("_")
                    metric = approach_name(metric, param=param, cam=False)
                order = np.argsort(-arr)
                orders[(metric, model_id)] = order

    apfds = dict()

    for i in range(FIRST_K_MODELS_CONSIDERED):
        for approach in APPROACHES:
            try:
                order = orders[(approach, str(i))]
                m = misclassifications[str(i)]
            except KeyError:
                if not approach == "VR" and case_study == "cifar10":
                    warnings.warn(f"missing results for {approach} on {case_study}")
                continue

            apfd = apfd_from_order(m, order)
            try:
                apfds[approach][i] = apfd
            except KeyError:
                apfds[approach] = dict()
                apfds[approach][i] = apfd

    return apfds


def _get_as_df(case_studies) -> pd.DataFrame:
    col_idx = pd.MultiIndex.from_product([case_studies, ["nominal", "ood", TIME_COL]])

    category_and_rows = [_row(row) for row in APPROACHES]
    row_index = pd.MultiIndex.from_tuples(
        category_and_rows, names=["category", "approach"]
    )
    df = pd.DataFrame(columns=col_idx, index=row_index)

    for case_study in case_studies:
        for ds in ["nominal", "ood"]:
            print(f"Working on {case_study} {ds}")
            apfds = load_apfd_values(case_study, ds)
            for category, approach in category_and_rows:
                try:
                    df.loc[(category, approach), (case_study, ds)] = np.mean(
                        list(apfds[approach].values())
                    )
                except KeyError:
                    df.loc[(category, approach), (case_study, ds)] = "n.a."
    return df


def _plot_latex_table(pd_df: pd.DataFrame):
    """Plots a nice latex table from the given pandas dataframe."""
    pd_df = pd_df.iloc[pd_df.index.get_level_values("approach").isin(PAPER_APPROACHES)]

    pd_df.rename(mapper=human_appraoch_name, axis="index", inplace=True)

    latex = pd_df.to_latex(
        multicolumn_format="c",
        multirow=True,
        column_format="llcccccccccccc",
        float_format="{:.2%}".format,
    )

    # Format categories vertically
    latex = vertical_categories(latex)

    # Remove category header
    latex = latex.replace("category", "", 1)

    # Refer to replication package for full table
    skipped_tip = len(APPROACHES) - len(PAPER_APPROACHES)
    latex = latex.replace(
        "\\bottomrule",
        """
        \\bottomrule
        \multicolumn{14}{c}{\\begin{tabular}{@{}c@{}}
        \\footnotesize This table provides an overview of selected results.
        """
        + f"Find the full table, including {skipped_tip} additional TIP, as CSV, in the \\replipkg. "
        + """
        \end{tabular}}\\\\
        \\bottomrule
        """.replace(
            "\\t", ""
        ),
        1,
    )

    with open(f"{OUTPUT_FOLDER}/results/apfd_paper_table.tex", "w") as text_file:
        text_file.write(latex)


def _add_reported_times(df: pandas.DataFrame, partial_times: Dict[str, List[float]]):
    """Adds timing information to the given dataframe.

    Specifically, we track the
        - setup time (the time to 'prepare' the method on the training data, e.g. fitting KDE in LSA)
        - prediction time (the time to get predictin (and if applicable activations)
        - quantification time (the time to run the actual TIP on the observed activations)
        - cam time (optionally, the time to run cam on the TIP outputs)

    """
    # Equal than or less because of reproductions which may run less models
    assert (
        int(max(k[2] for k in partial_times.keys())) <= 9
    ), "Should Only consider first 10 runs"

    tips = set([(k[3], k[4]) for k in partial_times.keys()])
    case_studies = set([k[0] for k in partial_times.keys()])
    # tips contains all non-cam TIPs
    for cs in case_studies:
        for tc, tn in tips:

            def _match_k(k):
                return k[0] == cs and k[3] == tc and k[4] == tn

            if not any(_match_k(k) for k in partial_times.keys()):
                assert (
                    cs == "cifar10" and tc == "VR"
                ), "Time should only be missing for VR on CIFAR10"
                df.loc[_row("VR"), (cs, TIME_COL)] = "n.a."
                continue

            avg_pred_time = mean(
                [v[1] for k, v in partial_times.items() if _match_k(k)]
            )
            avg_quant_time = mean(
                [v[2] for k, v in partial_times.items() if _match_k(k)]
            )
            avg_cam_time = mean([v[3] for k, v in partial_times.items() if _match_k(k)])
            avg_setup_time = mean(
                [v[0] for k, v in partial_times.items() if _match_k(k)]
            )

            row = _times_naming_to_table_row(tc, tn)

            def _format_time(t):
                return f"{round(t)}s"

            # The total time is calculated as the average of the setup time,
            # the prediction time (for ood and nominal), and the quantification time (again, ood and nominal).
            # This is equivalent to avg(setup-time) + 2x avg(prediction-time) + 2x avg(quantification-time).
            non_cam_time = avg_setup_time + 2 * (avg_pred_time + avg_quant_time)
            df.loc[row, (cs, TIME_COL)] = _format_time(non_cam_time)

            if row[0] == "surprise" or row[0] == "neuron coverage":
                row = row[0], f"{row[1]}-cam"
                cam_time = non_cam_time + 2 * avg_cam_time
                df.loc[row, (cs, TIME_COL)] = _format_time(cam_time)


def _times_naming_to_table_row(tip_type: str, param: str):
    """Converts the TIP naming used in the timing artifacts to table-row indexes."""
    tip_type = "softmax" if tip_type == "SM" else tip_type
    tip_type = "softmax_entropy" if tip_type == "SE" else tip_type
    tip_type = "pcs" if tip_type == "PCS" else tip_type
    tip_type = "deep_gini" if tip_type == "DeepGini" else tip_type
    if param != "":
        tip_type = f"{tip_type}_{param}"
    return _row(tip_type)


def _fill_times(df):
    times = times_collector.load_times()
    _add_reported_times(df, times)
    return df


def run():
    """Runs this module. Called by this module main method, or reproduction CLI."""
    df = _get_as_df(["mnist", "fmnist", "cifar10", "imdb"])

    df = _fill_times(df)
    df.to_csv(f"{OUTPUT_FOLDER}results/apfds.csv")
    _plot_latex_table(df)


if __name__ == "__main__":
    run()
