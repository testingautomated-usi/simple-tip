import numpy as np
import tensorflow as tf

from src.dnn_test_prio.handler_model import BaseModel


def test_get_activations():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, activation="relu", input_shape=(20,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(
        tf.keras.layers.Dense(
            3,
            activation="relu",
        )
    )
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Softmax())

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    y = tf.one_hot(np.arange(200) % 4, 4)
    x = np.random.random((200, 20))
    model.fit(x, y, epochs=30, batch_size=50)

    base_model = BaseModel(model, [0, 2, 3])
    x = np.random.random((10, 20))
    ds = tf.data.Dataset.from_tensor_slices(x).batch(32)
    predictions_and_unc = base_model.get_pred_and_uncertainty(ds)
    activations = base_model.get_activations(x)

    # Check all inner layers are present
    assert len(activations) == 3
    # Check point predictions are the same in both functional and sequential model
    assert np.all(np.argmax(activations[-1], axis=1) == predictions_and_unc[0])
