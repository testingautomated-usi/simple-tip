"""Runner class for the Fashion-Mnist case study."""

import logging
import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import uncertainty_wizard

from src.dnn_test_prio import (
    activation_persistor,
    eval_active_learning,
    eval_prioritization,
)
from src.dnn_test_prio.case_study import CaseStudy

CASE_STUDY = "fmnist"

uncertainty_wizard.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()

SA_ACTIVATION_LAYERS = [3]

NC_ACTIVATION_LAYERS = [0, 1, 2, 3]

MNIST_PREDICTION_BADGE_SIZE = 128


def _train_new_model(x, y) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Train model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(x, y, batch_size=128, epochs=15, validation_split=0.1)
    return model


def _fashion_mnist_model_creator(model_id: int) -> Tuple[tf.keras.Model, any]:
    """
    Creates a model for the MNIST dataset.
    Taken from https://keras.io/examples/vision/mnist_convnet/
    """
    # prepare data
    (x_train, y_train), _, _ = _load_datasets()
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    model = _train_new_model(x_train, y_train)
    return model, None


def _fashion_mnist_active_learning_evaluator(
    model_id: int, model: tf.keras.Model
) -> None:
    # Verbose logging
    logging.basicConfig(level=logging.INFO)

    # Data Preparation
    (x_train, y_train), (x_test, y_test), (ood_x_test, ood_y_test) = _load_datasets()

    return eval_active_learning.evaluate(
        case_study=CASE_STUDY,
        model_id=model_id,
        model=model,
        train_x=x_train,
        train_y=y_train,
        nominal_test_x=x_test,
        nominal_test_labels=y_test,
        ood_test_x=ood_x_test,
        ood_test_labels=ood_y_test,
        nc_activation_layers=NC_ACTIVATION_LAYERS,
        sa_activation_layers=SA_ACTIVATION_LAYERS,
        training_process=_train_new_model,
        observed_share=0.5,
        num_selected=1000,
        num_classes=10,
    )


def _fashion_mnist_prio_evaluator(model_id: int, model: tf.keras.Model) -> None:
    # Verbose logging
    logging.basicConfig(level=logging.INFO)

    # Dataset Preparation
    (x_train, _), (x_test, y_test), (ood_x_test, ood_y_test) = _load_datasets()
    train = tf.data.Dataset.from_tensor_slices(x_train).batch(
        MNIST_PREDICTION_BADGE_SIZE
    )
    nominal_test_x = tf.data.Dataset.from_tensor_slices(x_test).batch(
        MNIST_PREDICTION_BADGE_SIZE
    )
    ood_test_x = tf.data.Dataset.from_tensor_slices(ood_x_test).batch(
        MNIST_PREDICTION_BADGE_SIZE
    )

    return eval_prioritization.evaluate(
        case_study=CASE_STUDY,
        model_id=model_id,
        model=model,
        training_dataset=train,
        nominal_test_dataset=nominal_test_x,
        nominal_test_labels=y_test,
        ood_test_dataset=ood_test_x,
        ood_test_labels=ood_y_test,
        nc_activation_layers=NC_ACTIVATION_LAYERS,
        sa_activation_layers=SA_ACTIVATION_LAYERS,
    )


def _fmnist_activation_persistor(model_id: int, model: tf.keras.Model) -> None:
    logging.basicConfig(level=logging.INFO)
    train, nom, ood = _load_datasets()
    return activation_persistor.persist(
        model=model,
        case_study="fmnist",
        model_id=model_id,
        train_set=train,
        test_nominal=nom,
        test_corrupted=ood,
    )


def _load_datasets():
    # prepare data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    ood_x_test = np.load("../../datasets/fmnist-c-test.npy").astype("float32") / 255.0
    ood_y_test = np.load("../../datasets/fmnist-c-test-labels.npy")

    ood_x_test = np.concatenate((x_test, np.expand_dims(ood_x_test, -1)), axis=0)
    ood_y_test = np.concatenate((y_test, ood_y_test), axis=0)
    shuffle_args = np.random.default_rng(0).permutation(len(ood_y_test))
    ood_x_test = ood_x_test[shuffle_args]
    ood_y_test = ood_y_test[shuffle_args]
    return (x_train, y_train), (x_test, y_test), (ood_x_test, ood_y_test)


class FashionMnistCaseStudy(CaseStudy):
    """Utility class to run Fashion Mnist experiments."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _prefetch_datasets():
        # Cache nominal mnist dataset
        tf.keras.datasets.fashion_mnist.load_data()
        # Make sure OOD dataset is cached
        assert os.path.exists("../../datasets/fmnist-c-test.npy")
        assert os.path.exists("../../datasets/fmnist-c-test-labels.npy")

    # docstr-coverage: inherited
    def get_name(self):
        return CASE_STUDY

    def _model_creator(self) -> Callable[[int], Tuple[tf.keras.Model, any]]:
        return _fashion_mnist_model_creator

    @staticmethod
    def _prioritization_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _fashion_mnist_prio_evaluator

    @staticmethod
    def _active_learning_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _fashion_mnist_active_learning_evaluator

    @staticmethod
    def _activation_persistor() -> Callable[[int, tf.keras.Model], None]:
        return _fmnist_activation_persistor


if __name__ == "__main__":
    cs = FashionMnistCaseStudy()

    models = 10
    # cs.train(list(range(models)), num_processes=0)
    cs.run_prio_eval(list(range(models)), num_processes=0)
    # cs.run_active_learning_eval(list(range(models)), num_processes=0)
