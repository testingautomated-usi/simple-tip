"""Runner class for the IMDB case study."""

import logging
import os.path
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import uncertainty_wizard

from datasets import load_dataset
from src.core.text_corruptor import TextCorruptor
from src.dnn_test_prio import (
    activation_persistor,
    eval_active_learning,
    eval_prioritization,
    memory_leak_avoider,
)
from src.dnn_test_prio.case_study import OUTPUT_FOLDER, CaseStudy

CASE_STUDY = "imdb"

VOCAB_SIZE = 2000

INPUT_MAXLEN = 100

DS_CACHE_FOLDER = os.path.join(OUTPUT_FOLDER, ".external_datasets", "imdb")
HUGGINGFACE_CACHE_DIR = os.path.join(DS_CACHE_FOLDER, "huggingface")

uncertainty_wizard.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()

SA_ACTIVATION_LAYERS = [5]

NC_ACTIVATION_LAYERS = [
    (1, lambda x: x.token_emb),  # Embedding layers
    (1, lambda x: x.pos_emb),  # Embedding layers
    (2, lambda x: x.ffn[0]),  # Dense feed forward layers in transformer
    (2, lambda x: x.ffn[1]),  # Dense feed forward layers in transformer
    3,
    5,  # Dense layers in classifier
]

BADGE_SIZE = 128


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer layer."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = (
            embed_dim,
            num_heads,
            ff_dim,
            rate,
        )

    # docstr-coverage: inherited
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    # docstr-coverage: inherited
    def get_config(self):
        config = super().get_config().copy()
        config["embed_dim"] = self.embed_dim
        config["num_heads"] = self.num_heads
        config["ff_dim"] = self.ff_dim
        config["rate"] = self.rate
        config["att"] = self.att.get_config()
        config["ffn"] = self.ffn.get_config()
        config["layernorm1"] = self.layernorm1.get_config()
        config["layernorm2"] = self.layernorm2.get_config()
        config["dropout1"] = self.dropout1.get_config()
        config["dropout2"] = self.dropout2.get_config()
        return config

    # docstr-coverage: inherited
    @classmethod
    def from_config(cls, config):
        instance = cls(
            config["embed_dim"], config["num_heads"], config["ff_dim"], config["rate"]
        )
        instance.att = tf.keras.layers.MultiHeadAttention.from_config(config["att"])
        instance.ffn = tf.keras.Sequential.from_config(config["ffn"])
        instance.layernorm1 = tf.keras.layers.LayerNormalization.from_config(
            config["layernorm1"]
        )
        instance.layernorm2 = tf.keras.layers.LayerNormalization.from_config(
            config["layernorm2"]
        )
        instance.dropout1 = tf.keras.layers.Dropout.from_config(config["dropout1"])
        instance.dropout2 = tf.keras.layers.Dropout.from_config(config["dropout2"])
        return instance


# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class MyTokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Construct the embedding matrix."""

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        # super(MyTokenAndPositionEmbedding, self).__init__()
        super(MyTokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim

    # docstr-coverage: inherited
    def build(self, input_shape):
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim
        )
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim
        )

    # docstr-coverage: inherited
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    # docstr-coverage: inherited
    def get_config(self):
        return {
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }


def _train_new_model(x_train, y_train) -> tf.keras.Model:
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(INPUT_MAXLEN,))
    embedding_layer = MyTokenAndPositionEmbedding(INPUT_MAXLEN, VOCAB_SIZE, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # As opposed to the keras tutorial, we use categorical_crossentropy,
    #   and we run 10 instead of 2 epochs, but with early stopping.
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        # callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    return model


def _imdb_model_creator(model_id: int) -> Tuple[tf.keras.Model, any]:
    """
    Creates a model for the MNIST dataset.
    Taken from https://keras.io/examples/vision/mnist_convnet/
    """
    # prepare data
    (x_train, y_train), _, _ = _load_datasets()
    y_train = tf.keras.utils.to_categorical(y_train)
    model = _train_new_model(x_train, y_train)
    return model, None


def _imdb_active_learning_evaluator(model_id: int, model: tf.keras.Model) -> None:
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
        num_selected=2500,
        num_classes=2,
        dsa_badge_size=500,  # Override default badge size
    )


def _imdb_prio_evaluator(model_id: int, model: tf.keras.Model) -> None:
    # Verbose logging
    logging.basicConfig(level=logging.INFO)

    # set prediction badge size
    # Attention: Don't rename. This will be attempted to be read when predicting,
    #   failing silently if not set.
    model.custom_badge_size = 600

    # Dataset Preparation
    (x_train, _), (x_test, y_test), (ood_x_test, ood_y_test) = _load_datasets()
    train = tf.data.Dataset.from_tensor_slices(x_train).batch(BADGE_SIZE)
    nominal_test_x = tf.data.Dataset.from_tensor_slices(x_test).batch(
        model.custom_badge_size
    )
    ood_test_x = tf.data.Dataset.from_tensor_slices(ood_x_test).batch(
        model.custom_badge_size
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
        dsa_badge_size=500,  # Override default badge size
    )


def _imdb_activation_persistor(model_id: int, model: tf.keras.Model) -> None:
    logging.basicConfig(level=logging.INFO)
    train, nom, ood = _load_datasets()
    return activation_persistor.persist(
        model=model,
        case_study="imdb",
        model_id=model_id,
        train_set=train,
        test_nominal=nom,
        test_corrupted=ood,
    )


def _load_datasets():
    x_train = np.load(os.path.join(DS_CACHE_FOLDER, "x_train.npy"))
    y_train = np.load(os.path.join(DS_CACHE_FOLDER, "y_train.npy"))
    x_test = np.load(os.path.join(DS_CACHE_FOLDER, "x_test.npy"))
    y_test = np.load(os.path.join(DS_CACHE_FOLDER, "y_test.npy"))
    corrupted_x_test = np.load(os.path.join(DS_CACHE_FOLDER, "x_corrupted.npy"))

    ood_x_test = np.concatenate((x_test, corrupted_x_test), axis=0)
    ood_y_test = np.concatenate((y_test, y_test), axis=0)
    # Shuffle ood_x_test and ood_y_test
    perm = np.random.permutation(len(ood_x_test))
    ood_x_test = ood_x_test[perm]
    ood_y_test = ood_y_test[perm]

    return (x_train, y_train), (x_test, y_test), (ood_x_test, ood_y_test)


class ImdbCaseStudy(CaseStudy):
    """Utility class to run IMDB experiments."""

    def __init__(self):
        super().__init__()

    # docstr-coverage: inherited
    @staticmethod
    def _prefetch_datasets():
        if (
            os.path.exists(os.path.join(DS_CACHE_FOLDER, "x_train.npy"))
            and os.path.exists(os.path.join(DS_CACHE_FOLDER, "y_train.npy"))
            and os.path.exists(os.path.join(DS_CACHE_FOLDER, "x_test.npy"))
            and os.path.exists(os.path.join(DS_CACHE_FOLDER, "y_test.npy"))
            and os.path.exists(os.path.join(DS_CACHE_FOLDER, "x_corrupted.npy"))
        ):
            return

        train_ds = load_dataset("imdb", cache_dir=HUGGINGFACE_CACHE_DIR, split="train")
        x_train, y_train = train_ds["text"], train_ds["label"]

        test_ds = load_dataset("imdb", cache_dir=HUGGINGFACE_CACHE_DIR, split="test")
        x_test, y_test = test_ds["text"], test_ds["label"]

        all_x = load_dataset(
            "imdb", cache_dir=HUGGINGFACE_CACHE_DIR, split="train+test"
        )["text"]

        corruptor = TextCorruptor(
            base_dataset=all_x, cache_dir=os.path.join(DS_CACHE_FOLDER, "corruptor")
        )
        x_test_ood = corruptor.corrupt(x_test, severity=0.5, seed=0)

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_corrupted = tokenizer.texts_to_sequences(x_test_ood)

        x_train = tf.keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=INPUT_MAXLEN
        )
        x_test = tf.keras.preprocessing.sequence.pad_sequences(
            x_test, maxlen=INPUT_MAXLEN
        )
        x_corrupted = tf.keras.preprocessing.sequence.pad_sequences(
            x_corrupted, maxlen=INPUT_MAXLEN
        )

        if not os.path.exists(DS_CACHE_FOLDER):
            os.makedirs(DS_CACHE_FOLDER)
        np.save(os.path.join(DS_CACHE_FOLDER, "x_train.npy"), x_train)
        np.save(os.path.join(DS_CACHE_FOLDER, "y_train.npy"), y_train)
        np.save(os.path.join(DS_CACHE_FOLDER, "x_test.npy"), x_test)
        np.save(os.path.join(DS_CACHE_FOLDER, "y_test.npy"), y_test)
        np.save(os.path.join(DS_CACHE_FOLDER, "x_corrupted.npy"), x_corrupted)

    # docstr-coverage: inherited
    def get_name(self):
        return "imdb"

    def _model_creator(self) -> Callable[[int], Tuple[tf.keras.Model, any]]:
        return _imdb_model_creator

    @staticmethod
    def _prioritization_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _imdb_prio_evaluator

    @staticmethod
    def _active_learning_evaluator() -> Callable[[int, tf.keras.Model], None]:
        return _imdb_active_learning_evaluator

    @staticmethod
    def _activation_persistor() -> Callable[[int, tf.keras.Model], None]:
        return _imdb_activation_persistor


if __name__ == "__main__":
    cs = ImdbCaseStudy()

    cs.train(
        list(range(100)), num_processes=3, context=memory_leak_avoider.SingleUseContext
    )
    cs.run_prio_eval(list(range(10)), num_processes=0)
    cs.run_prio_eval(
        list(range(10, 100)),
        num_processes=3,
        context=memory_leak_avoider.SingleUseContext,
    )
    cs.run_active_learning_eval(
        list(range(100)), num_processes=2, context=memory_leak_avoider.SingleUseContext
    )
