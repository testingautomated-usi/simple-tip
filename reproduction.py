"""This is the main entrypoint when interactively running the reproduction package."""
import os
from enum import Enum
from typing import List, Optional, Union

import click
import typer

from src.dnn_test_prio.case_study import OUTPUT_FOLDER


class ReproductionType(str, Enum):
    """The type of work to reproduce."""

    TRAINING = "training"
    TEST_PRIO = "test_prio"
    ACTIVE_LEARNING = "active_learning"
    EVAL = "evaluation"
    ACTIVATION_COLLECTION = "at_collection"


class CaseStudyType(str, Enum):
    """The type of case study to reproduce."""

    MNIST = "mnist"
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fmnist"
    IMDB = "imdb"


class EvalType(str, Enum):
    """The type of evaluation to reproduce."""

    TEST_PRIO = "test_prio"
    ACTIVE_LEARNING = "active_learning"
    APFD_STATS = "test_prio_statistics"
    ACTIVE_STATS = "active_learning_statistics"


REPR_TYPE_PROMPT = "Please select the type of work to reproduce"


def _check_ready_to_start(message: str):
    typer.echo(f"\n {message} \n")
    typer.confirm("Are you sure you want to start?", default=True, abort=True)


def _setup_eval():
    typer.echo(
        """
        You can choose between the following evaluation types:
        - test_prio: Evaluates the test prioritization (table 1 in paper and its extended csv version)
        - active_learning: Evaluates the active learning (table 2 in paper and its extended csv version)
        - test_prio_statistics: Statistics about the test prioritization (fig 3 in paper and p-val/eff-size csv)
        - active_learning_statistics: Statistics about the active learning (fig 4 in paper and p-val/eff-size csv)
        """
    )
    eval: CaseStudyType = typer.prompt(
        "Which outcome do you want to reproduce?",
        type=click.Choice([c.value for c in EvalType], case_sensitive=False),
    )

    _check_ready_to_start(
        "This will override some results in the `assets/results` folder. "
        "If you want, you can delete the corresponding files (but not the folder) now, "
        "to verify that they are indeed re-generated."
    )

    if eval == EvalType.TEST_PRIO.value:
        from src.plotters import eval_apfd_table

        eval_apfd_table.run()
    elif eval == EvalType.ACTIVE_LEARNING.value:
        from src.plotters import eval_active_learning_table

        eval_active_learning_table.run()
    elif eval == EvalType.APFD_STATS.value:
        from src.plotters import eval_apfd_correlation

        eval_apfd_correlation.run()
    elif eval == EvalType.ACTIVE_STATS.value:
        from src.plotters import eval_active_correlation

        eval_active_correlation.run()
    else:
        raise ValueError(f"Unknown eval type: {eval}")

    typer.echo(
        "Done. Check your /assets/results folder to find the reproduced result files."
    )


def _cs_runner_for_case_study(case_study: CaseStudyType):
    if case_study == CaseStudyType.MNIST:
        from src.dnn_test_prio.case_study_mnist import MnistCaseStudy

        return MnistCaseStudy()
    elif case_study == CaseStudyType.CIFAR10:
        from src.dnn_test_prio.case_study_cifar10 import Cifar10CaseStudy

        return Cifar10CaseStudy()
    elif case_study == CaseStudyType.FASHION_MNIST:
        from src.dnn_test_prio.case_study_fashion_mnist import FashionMnistCaseStudy

        return FashionMnistCaseStudy()
    elif case_study == CaseStudyType.IMDB:
        from src.dnn_test_prio.case_study_imdb import ImdbCaseStudy

        return ImdbCaseStudy()
    else:
        raise ValueError(f"Unknown case study: {case_study}")


def _setup_non_eval(r_type: ReproductionType):
    # Make sure user really wants to do this
    if r_type == ReproductionType.ACTIVATION_COLLECTION:
        typer.echo(
            f"Note that activation collection has only been added after paper"
            f"publication (due to a 3rd party request). "
            f"This step has thus not been used to collect our results, "
            f"but may help you when doing your own activation-based work."
        )

    confirmed = typer.confirm(
        f"Are you sure you want to run the {r_type} steps of the experiments? "
        f"These typically take a long time to run, "
        f"are not fully deterministic and will override files in your assets directory."
    )

    if not confirmed:
        typer.echo(f"Understood. Try running `{ReproductionType.EVAL.value}` instead")
        raise typer.Abort()

    case_study: CaseStudyType = typer.prompt(
        "Please enter the case study you want to run",
        type=click.Choice([c.value for c in CaseStudyType], case_sensitive=False),
    )
    run: Union[int, List[int]] = typer.prompt(
        "Please enter the run(s) you want to reproduce "
        "(choose -1 for all runs) [-1, 0-99] ",
        type=int,
    )

    if run == -1:
        typer.confirm(
            f"Are you sure you want to reproduce all 100 runs for this {case_study}? "
            "This is likely to take days or even weeks to run, dependent on your hardware",
            default=False,
            abort=True,
            err=True,
        )
        run = list(range(100))
    else:
        run = [run]

    _check_ready_to_start(
        f"Ok. We're good to go. After this, even if you abort, you may change or even "
        f"break the original, intermediate results stored in the `assets` folder. "
        f"To restore, re-download it from zenodo. Ready?"
    )

    cs_runner = _cs_runner_for_case_study(case_study)

    from src.dnn_test_prio.memory_leak_avoider import SingleUseContext

    if r_type == ReproductionType.TRAINING:
        cs_runner.train(run, num_processes=1, context=SingleUseContext)
    elif r_type == ReproductionType.TEST_PRIO:
        cs_runner.run_prio_eval(run, num_processes=1, context=SingleUseContext)
    elif r_type == ReproductionType.ACTIVE_LEARNING:
        cs_runner.run_active_learning_eval(
            run, num_processes=1, context=SingleUseContext
        )
    elif r_type == ReproductionType.ACTIVATION_COLLECTION:
        cs_runner.collect_activations(
            model_ids=run, num_processes=1, context=SingleUseContext
        )
    else:
        typer.echo(f"Unknown reproduction type: {r_type}", err=True)

    typer.echo("Done.")


def main(
    phase: Optional[ReproductionType] = typer.Option(
        default="evaluation", prompt=REPR_TYPE_PROMPT
    )
):
    """This is the main entrypoint when running the reproduction package."""

    assert os.path.exists(OUTPUT_FOLDER), (
        f"Assets directory is not mounted to {OUTPUT_FOLDER}. "
        f"Please check the reproduction package documentation for explanations"
        f"on how to mount the assets folder."
    )

    if phase == ReproductionType.EVAL:
        _setup_eval()
    else:
        _setup_non_eval(phase)


if __name__ == "__main__":
    typer.run(main)
