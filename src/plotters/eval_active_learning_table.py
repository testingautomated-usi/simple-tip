"""Evaluates active learning performances (table 2)."""

import re
import warnings
from typing import Dict, List, Tuple

import pandas as pd

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.plotters.utils import (
    APPROACHES,
    PAPER_APPROACHES,
    _row,
    human_appraoch_name,
    load_all_for_regex,
    vertical_categories,
)

BASELINE = "random"

ORIGINAL = "original_na.pickle"

RANDOM = "random"


def load_arrays_active_learning(
    case_study: str, ds_name: str, by_id: bool = False
) -> Dict[str, List[Dict[Tuple[str, str], float]]]:
    """Loads  active learning per-run raw results for a given case study and dataset."""
    res = dict()
    incl_random = APPROACHES.copy()
    incl_random.append(RANDOM)
    for approach in incl_random:
        file_name_format = f"{case_study}_\d*_{approach}_{ds_name}\."
        regex = re.compile(
            file_name_format.format(
                case_study=case_study, approach_name=approach, ds_name=ds_name
            )
        )
        vals, files = load_all_for_regex("active_learning", regex)
        if not by_id:
            res[approach] = vals
        else:
            res[approach] = {
                int(files[i].split("_")[1]): vals[i] for i in range(len(vals))
            }

    original_file_name = f"{case_study}_\d*_original_na\."
    original_vals, original_files = load_all_for_regex(
        "active_learning", re.compile(original_file_name)
    )
    if not by_id:
        res["original"] = original_vals
    else:
        res["original"] = {
            int(original_files[i].split("_")[1]): original_vals[i]
            for i in range(len(original_vals))
        }
    return res


def _reduce_active_learning(
    cs: str, active_learning_files: Dict[str, List[Dict[Tuple[str, str], float]]]
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Reduce results of multiple runs into a single result."""
    res = dict()
    for approach, run_results in active_learning_files.items():
        # Sanity check to make sure all runs have the same keys
        approach_res = dict()
        assert all(
            run_results[0].keys() == run_results[i].keys()
            for i in range(1, len(run_results))
        )
        # Take average of all runs
        if len(run_results) == 0:
            if not (approach == "VR" and cs == "cifar10"):
                warnings.warn("missing results")
            continue
        for key in run_results[0].keys():
            avg = sum(run_results[i][key] for i in range(len(run_results))) / len(
                run_results
            )
            approach_res[key] = avg
        res[approach] = approach_res
    return res


def _relative_active_learning_gains(
    reduced_performances: Dict[str, Dict[Tuple[str, str], float]], baseline: str
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Performance of TIP active-learning model - performance of random active-learning model."""
    assert baseline in ["random", "original"]
    assert baseline in reduced_performances.keys()
    res = dict()
    for approach, performance in reduced_performances.items():
        if approach == baseline:
            continue
        res[approach] = dict()
        for key in performance.keys():
            res[approach][key] = performance[key] - reduced_performances[baseline][key]
    return res


def build_data_frame(case_studies: List[str]):
    """Build a pandas dataframe containing the active learning results"""
    col_idx = pd.MultiIndex.from_product(
        [
            case_studies,
            ["nominal", "ood"],
            [
                "nominal:observed",
                "nominal:future",
                "ood:observed",
                "ood:future",
            ],
        ]  # , names=["case_study", "active split", "eval split"]
    )

    rows = ["original", "random"]
    rows.extend(APPROACHES)
    category_and_rows = [_row(row) for row in rows]
    row_index = pd.MultiIndex.from_tuples(
        category_and_rows, names=["category", "approach"]
    )
    df = pd.DataFrame(columns=col_idx, index=row_index)

    for cs in case_studies:
        for obs in ["nominal", "ood"]:
            print(f"Loading {cs}_{obs}")
            file_values = load_arrays_active_learning(cs, obs)
            reduced = _reduce_active_learning(cs, file_values)
            relative = _relative_active_learning_gains(reduced, BASELINE)
            for approach in ["original", "random"]:
                for key in reduced[approach].keys():
                    df.at[_row(approach), (cs, obs, f"{key[0]}:{key[1]}")] = _forma(
                        reduced[approach][key]
                    )

            for approach in APPROACHES:
                try:
                    for key in relative[approach].keys():
                        df.at[_row(approach), (cs, obs, f"{key[0]}:{key[1]}")] = _forma(
                            relative[approach][key]
                        )
                except KeyError:
                    df.at[_row(approach), (cs, obs, f"nominal:observed")] = "n.a."
                    df.at[_row(approach), (cs, obs, f"nominal:future")] = "n.a."
                    df.at[_row(approach), (cs, obs, f"ood:observed")] = "n.a."
                    df.at[_row(approach), (cs, obs, f"ood:future")] = "n.a."

    return df


def _forma(x):
    return "{:.2%}".format(x)


def latex_table(pd_df: pd.DataFrame):
    """Creates a nice latex table from a given active-learning results dataframe"""
    # Filter to contain only the TIP which are shown in the paper
    #   (subselection done for reasons of space, CSV will contain all rows)
    paper_approaches = PAPER_APPROACHES.copy()
    paper_approaches.extend(["original", "random"])
    pd_df = pd_df.iloc[pd_df.index.get_level_values("approach").isin(paper_approaches)]

    pd_df.rename(mapper=human_appraoch_name, axis="index", inplace=True)

    paper_columns = []
    for column in pd_df.columns:
        if column[2].startswith(column[1]) and column[2].endswith("future"):
            paper_columns.append(column)
    latex = pd_df.to_latex(
        columns=paper_columns,
        multicolumn_format="c",
        multirow=True,
        column_format="llcccccccc",
    )

    # Put note after baselines explaining the future results are differences
    latex = latex.replace(
        "\cline{1-10}",
        """
                          &&\multicolumn{8}{c}{\\footnotesize{Subsequent results are differences to the \emph{"""
        + BASELINE
        + """} baseline.}}\\\\
                          \cline{1-10}
                          """.replace(
            "\\t", ""
        ),
        1,
    )

    # Remove row showing eval split, as we only consider the one which
    #   corresponds to the active split
    latex = "\n".join([i for i in latex.split("\n") if "nominal:future" not in i])

    # Format categories vertically
    latex = vertical_categories(latex)

    # Remove category header
    latex = latex.replace("category", "", 1)

    # Refer to replication package for full table
    skipped_tip = (
        len(APPROACHES) - len(paper_approaches) + 2
    )  # +2 for original and random
    latex = latex.replace(
        "\\bottomrule",
        """
        \\bottomrule
        \multicolumn{10}{c}{\\begin{tabular}{@{}c@{}}
        \\footnotesize This table provides an overview of selected results.\\\\
        """
        + f"\\footnotesize Find the full table, including {skipped_tip} additional TIP, cross dataset (nominal/ood) evaluations \\\\  "
        + """
        \\footnotesize and evaluations on the active splits used to select inputs for retraining as CSV in the \\replipkg.
        \end{tabular}}\\\\
        \\bottomrule
        """.replace(
            "\\t", ""
        ),
        1,
    )
    with open(f"{OUTPUT_FOLDER}/results/active_paper_table.tex", "w") as text_file:
        text_file.write(latex)


def run():
    """Runs this module. Called by this module main method, or reproduction CLI."""
    df = build_data_frame(["mnist", "fmnist", "cifar10", "imdb"])
    df.to_csv(f"{OUTPUT_FOLDER}/results/active.csv")
    latex_table(df)


if __name__ == "__main__":
    run()
