import numpy as np

from src.core.deepgini import DeepGini


def test_static_attributes():
    """Regression test. Checks that the static attributes are correct."""
    assert DeepGini.takes_samples() is False
    assert DeepGini.is_confidence() is False
    # See uncertainty wizard docs
    for alias in DeepGini.aliases():
        assert alias.startswith("custom")


def test_quantification():
    input_batch = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.1, 0.1, 0.3],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
        ]
    )

    expected = np.array(
        [
            0.7,  # https://bit.ly/301vmQ3
            0.64,  # https://bit.ly/3qkHuGm
            0.75,  # https://bit.ly/3wrPI0h
            0,  # Trivial
            0,  # Re-Ordering of previous
        ]
    )

    pred, unc = DeepGini.calculate(input_batch)
    assert np.all(pred == np.array([3, 0, 0, 0, 1]))
    assert np.all(unc == expected)
