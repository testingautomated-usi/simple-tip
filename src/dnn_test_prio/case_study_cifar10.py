"""Runner class for the Cifar10 case study."""

import logging
import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import uncertainty_wizard

from src.dnn_test_prio import eval_active_learning, eval_prioritization
from src.dnn_test_prio.case_study import OUTPUT_FOLDER, CaseStudy

CASE_STUDY = "cifar10"

uncertainty_wizard.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()

from uncertainty_wizard.models.ensemble_utils import DynamicGpuGrowthContextManager

SA_ACTIVATION_LAYERS = [3]

NC_ACTIVATION_LAYERS = [0, 1, 2, 3]

CIFAR_10_BATCH_SIZE = 32

ZENDO_PATH_ON_FS = f"{OUTPUT_FOLDER}/.external_datasets/CIFAR-10-C/"


def _train_new_model(x, y) -> tf.keras.Model:
    # Model Architecture taken from https://www.tensorflow.org/tutorials/images/cnn
    #   with the following changes:
    #   - 20 epochs instead of 10
    #   - softmax output layer instead of "from_logits" in loss function

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Train model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x, y, epochs=20, validation_split=0.1, batch_size=32)
    return model


def _cifar10_model_creator(model_id: int) -> Tuple[tf.keras.Model, any]:
    """
    Creates a model for the MNIST dataset.
    Taken from https://keras.io/examples/vision/mnist_convnet/
    """
    # prepare data
    (x_train, y_train), _, _ = _load_datasets()
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    model = _train_new_model(x_train, y_train)
    return model, None


def _mnist_active_learning_evaluator(model_id: int, model: tf.keras.Model) -> None:
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


def _cifar10_prio_evaluator(model_id: int, model: tf.keras.Model) -> None:
    # Verbose logging
    logging.basicConfig(level=logging.INFO)

    # Dataset Preparation
    (x_train, _), (x_test, y_test), (ood_x_test, ood_y_test) = _load_datasets()
    train = tf.data.Dataset.from_tensor_slices(x_train).batch(CIFAR_10_BATCH_SIZE)
    nominal_test_x = tf.data.Dataset.from_tensor_slices(x_test).batch(
        CIFAR_10_BATCH_SIZE
    )
    ood_test_x = tf.data.Dataset.from_tensor_slices(ood_x_test).batch(
        CIFAR_10_BATCH_SIZE
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


def _load_datasets():
    # prepare data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_test, y_train = y_test.flatten(), y_train.flatten()
    ood_x_test = (
        np.load("../../datasets/cifar10_c_images.npy").astype("float32") / 255.0
    )
    ood_y_test = np.load("../../datasets/cifar10_c_labels.npy")

    ood_x_test = np.concatenate((x_test, ood_x_test), axis=0)
    ood_y_test = np.concatenate((y_test.flatten(), ood_y_test), axis=0)
    shuffle_args = np.random.default_rng(0).permutation(len(ood_y_test))
    ood_x_test = ood_x_test[shuffle_args]
    ood_y_test = ood_y_test[shuffle_args]
    return (x_train, y_train), (x_test, y_test), (ood_x_test, ood_y_test)


class Cifar10CaseStudy(CaseStudy):
    """Utility class to run Cifar-10 experiments."""

    def __init__(self):
        super().__init__()

    # docstr-coverage: inherited
    @staticmethod
    def _prefetch_datasets():
        # Cache nominal dataset
        tf.keras.datasets.cifar10.load_data()
        # Cache corrupted dataset
        if not os.path.exists(
            "../../datasets/cifar10_c_images.npy"
        ) or not os.path.exists("../../datasets/cifar10_c_labels.npy"):
            if not os.path.exists(ZENDO_PATH_ON_FS):
                raise FileNotFoundError(
                    f"Zenodo dataset for cifar10-c not found."
                    f"Please download it from "
                    f"https://zenodo.org/record/2535967/files/CIFAR-10-C.tar "
                    f"and unpack it to to the .external_datasets folder"
                    f"in the mounted assets drive. \n"
                    f"Note that we cannot ship this with our replication package"
                    f"for copyright reasons."
                )
            else:
                logging.info("Loading CIFAR-10-C dataset from Zendo.")
                # Iterate of all files in ZENDO_PATH_ON_FS
                all_corruptions = []
                labels = None
                for file in os.listdir(ZENDO_PATH_ON_FS):
                    # Load the .npy file
                    if file == "labels.npy":
                        labels = np.load(os.path.join(ZENDO_PATH_ON_FS, file))
                    elif file.endswith(".npy"):
                        logging.info(f"Loading {file}")
                        all_corruptions.append(
                            np.load(os.path.join(ZENDO_PATH_ON_FS, file))
                        )
                # Concatenate all data
                num_corruptions = len(all_corruptions)
                all_corruptions = np.concatenate(all_corruptions, axis=0)
                # Choose 10000 random samples, over all corruptions and severities
                indexes = np.random.default_rng(0).permutation(len(all_corruptions))[
                    :10000
                ]
                images = all_corruptions[indexes]
                labels = np.tile(labels, num_corruptions)[indexes]

                np.save("../../datasets/cifar10_c_images.npy", images)
                np.save("../../datasets/cifar10_c_labels.npy", labels)

    # docstr-coverage: inherited
    def get_name(self):
        return "cifar10"

    def _model_creator(self) -> Callable[[int], Tuple[tf.keras.Model, any]]:
        return _cifar10_model_creator

    @staticmethod
    def _prioritization_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _cifar10_prio_evaluator

    @staticmethod
    def _active_learning_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _mnist_active_learning_evaluator


if __name__ == "__main__":
    cs = Cifar10CaseStudy()

    models = 10
    # cs.train(list(range(models)), num_processes=0)
    cs.run_prio_eval(list(range(models)), num_processes=0)
    # cs.run_active_learning_eval(list(range(models)), num_processes=0)
