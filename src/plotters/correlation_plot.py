"""Utilities to create the p-value and effect-size heatmap plots."""

from math import comb
from numbers import Number
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns

# Type Aliases
from matplotlib.colors import LogNorm

from src.dnn_test_prio.case_study import OUTPUT_FOLDER
from src.plotters.utils import human_approach_names

SAMPLE_KEY = Union[int, str]
APPROACH_KEY = Union[int, str]


def _paired_vargha_delanay_a12(x: List[float], y: List[float], paired=True) -> float:
    assert len(x) == len(y)
    # Unpaired implementation of A12
    x, y = np.array(x), np.array(y)
    if not paired:
        y = np.expand_dims(y, axis=1)
    same = np.sum(x == y)
    bigger = np.sum(x > y)
    a12 = (bigger + 0.5 * same) / (x == y).size

    return 2 * abs(a12 - 0.5)


class WilcoxonCorrelationPlot:
    """Class to create and modify a nice correlation plot."""

    def __init__(self, approaches: List[str], num_tested_approaches: int):
        self.p_value_calculator = lambda x, y: pg.wilcoxon(
            x, y, alternative="two-sided"
        )["p-val"][0]
        self.effect_size_calculator = lambda x, y: _paired_vargha_delanay_a12(x, y)
        self.error_correction = lambda p_values: p_values * comb(
            num_tested_approaches, 2
        )
        assert len(set(approaches)) == len(approaches), "Approach names must be unique"
        self.approaches = approaches
        self.measurements: Dict[APPROACH_KEY, Dict[SAMPLE_KEY, Number]] = {
            i: dict() for i in approaches
        }

    def add_measurement(
        self,
        approach: APPROACH_KEY,
        sample: SAMPLE_KEY,
        value: Number,
        unique: bool = True,
    ):
        """Registers an observation for statistical comparison"""
        if approach not in self.approaches:
            # Ignore if approach is not in the list of approaches
            return
        if unique:
            assert sample not in self.measurements[approach], (
                f"Sample key name must be unique for a given array. "
                f"Duplicate: {sample}."
                f"Pass `unqiue=False` to overwrite value."
            )
        self.measurements[approach][sample] = value

    def calc_values(self):
        """Calculates and caches the correlation values for all approaches"""
        grid_size = (len(self.approaches), len(self.approaches))
        res = dict()
        res["p"] = np.full(
            grid_size,
            10000,
            dtype=np.float,
        )
        res["e"] = np.full(
            grid_size,
            -10000,
            dtype=np.float,
        )
        res["num_samples"] = np.full(
            grid_size,
            -1000,
            dtype=np.int,
        )
        for i in range(len(self.approaches) - 1):
            for j in range(i + 1, len(self.approaches)):
                _, vals_i, val_j = self._common(i, j)
                res["num_samples"][i, j] = len(vals_i)
                if len(vals_i) == 0:
                    res["p"][i, j] = np.nan
                    res["e"][i, j] = np.nan
                elif val_j == vals_i:
                    res["p"][i, j] = np.nan
                    res["e"][i, j] = np.nan
                else:
                    res["p"][i, j] = self.p_value_calculator(vals_i, val_j)
                    res["e"][i, j] = self.effect_size_calculator(vals_i, val_j)
        return res

    def _common(self, i: int, j: int):
        """Finds the SAMPLE_KEYs which are shared by approaches for two approaches (by their order index)"""
        keys_1 = self.measurements[self.approaches[i]].keys()
        keys_2 = set(self.measurements[self.approaches[j]].keys())
        keys = set(keys_1).intersection(keys_2)

        values_1 = [self.measurements[self.approaches[i]][k] for k in keys]
        values_2 = [self.measurements[self.approaches[j]][k] for k in keys]

        return keys, values_1, values_2

    def plot_heatmap(self, exp: str, cs: str, ds: str):
        """Plots the heatmap of the correlation values"""
        #         https://newbedev.com/combining-two-heat-maps-in-seaborn
        values = self.calc_values()

        matrix_0 = np.triu(values["e"].transpose())
        error_corrected_p = self.error_correction(values["p"])
        matrix_1 = np.tril(error_corrected_p)

        ax_1 = sns.heatmap(
            values["e"].transpose(),
            annot=False,
            mask=matrix_0,
            cmap="inferno",
            square=True,
            cbar_kws=dict(
                shrink=0.6,
                pad=0.05,
                use_gridspec=True,
                location="bottom",
                label="Effect size",
            ),
        )
        ax_2 = sns.heatmap(
            values["p"],
            annot=False,
            mask=matrix_1,
            cmap="viridis",
            vmax=0.1,
            square=True,
            norm=LogNorm(),
            cbar_kws=dict(use_gridspec=True, location="right", label="P-Value"),
        )

        plt.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            labelbottom=False,
            bottom=False,
            top=True,
            labeltop=True,
        )

        # Show all ticks and label them with the respective list entries
        human_labels = human_approach_names(self.approaches)
        ax_2.set_xticks(
            np.arange(len(self.approaches)) + 0.5,
            labels=human_labels,
            rotation=45,
            ha="left",
        )
        ax_2.set_yticks(
            np.arange(len(self.approaches)) + 0.5, labels=human_labels, rotation=0
        )

        ax_1.hlines([3, 6], *ax_1.get_xlim(), color="white")
        ax_1.vlines([3, 6], *ax_1.get_ylim(), color="white")
        plt.axline((9, 9), (0, 0), linewidth=2, color="black")

        if cs != "all" or ds != "both":
            plt.savefig(
                f"{OUTPUT_FOLDER}/results/corr-{exp}-{cs}-{ds}.png", bbox_inches="tight"
            )
        else:
            plt.savefig(f"{OUTPUT_FOLDER}/results/corr-{exp}.png", bbox_inches="tight")

        plt.close()
