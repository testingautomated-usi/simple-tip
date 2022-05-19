"""Defines abstract components, to be implemented in each case study."""
import abc
from typing import Callable, List, Tuple

import tensorflow as tf
import uncertainty_wizard as uwiz
from uncertainty_wizard.models.ensemble_utils import EnsembleContextManager

MAX_NUM_MODELS = 100
OUTPUT_FOLDER = "/assets/"


class CaseStudy(abc.ABC):
    """Abstract class for a case study."""

    def __init__(self):
        self._prefetch_datasets()
        path = f"{OUTPUT_FOLDER}/models/{self.get_name()}"
        self.ens = uwiz.models.LazyEnsemble(
            num_models=MAX_NUM_MODELS,
            model_save_path=path,
            delete_existing=False,
            expect_model=True,
            default_num_processes=0,
        )

    @abc.abstractmethod
    def get_name(self):
        """The name of this case study (e.g., 'mnist', 'cifar10', ..)"""
        pass

    @staticmethod
    @abc.abstractmethod
    def _prefetch_datasets():
        """Downloads and prepares the datasets.

        Is called when constructing the case study instance,
        to make sure all datasets are ready on the file systems
        when experiments are started."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _model_creator() -> Callable[[int], Tuple[any, tf.keras.Model]]:
        """A picklable function to train new models, given the model id"""
        pass

    @staticmethod
    @abc.abstractmethod
    def _prioritization_evaluator() -> Callable[[int, tf.keras.Model], None]:
        """A picklable function to run test prioritization experiments.

        The returned function would typically be a wrapper around
        `eval_prioritization.evaluate(...)`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _active_learning_evaluator() -> Callable[[int, tf.keras.Model], None]:
        """A picklable function to run test active learning experiments.

        The returned function would typically be a wrapper around
        `eval_active_learning.evaluate(...)`."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _activation_persistor() -> Callable[[int, tf.keras.Model], None]:
        """A picklable function to save activation traces to file systems.

        The returned function would typically be a wrapper around
        `activation_persistor.persist(...)`.

        **This was not used for our experiments.**"""
        pass

    def train(
        self,
        model_ids: List[int],
        num_processes: int,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """Run the models training process.

        This is to be called by the case studies runners main method,
        or the reproduction CLI."""
        self.ens.create(
            self._model_creator(),
            models=model_ids,
            num_processes=num_processes,
            context=context,
        )

    def run_prio_eval(
        self,
        model_ids: List[int],
        num_processes: int,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """Run the models APFD evaluation experiments.

        This is to be called by the case studies runners main method,
        or the reproduction CLI."""
        self.ens.consume(
            self._prioritization_evaluator(),
            models=model_ids,
            num_processes=num_processes,
            context=context,
        )

    def run_active_learning_eval(
        self,
        model_ids: List[int],
        num_processes: int,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """Run the models active learning experiments.

        This is to be called by the case studies runners main method,
        or the reproduction CLI."""
        self.ens.consume(
            self._active_learning_evaluator(),
            models=model_ids,
            num_processes=num_processes,
            context=context,
        )

    def collect_activations(
        self,
        model_ids: List[int],
        num_processes: int,
        context: Callable[[int], EnsembleContextManager] = None,
    ):
        """Utility to store all activations on the file system.

        This was not actually used as part of our paper,
        but requested by a 3rd party to allow to build on our model
        activations without using our code directly."""
        self.ens.consume(
            self._activation_persistor(),
            models=model_ids,
            num_processes=num_processes,
            context=context,
        )
