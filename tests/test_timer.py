import time

import pytest

from src.core.timer import Timer


def test_timer_manual():
    """Test the timer under manual starting and stopping"""
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    timer.stop()
    assert 0.15 > timer.get() >= 0.1


def test_timer_context():
    """Test the timer when used as context wrapper."""
    timer = Timer()
    with timer:
        time.sleep(0.1)
    assert 0.15 > timer.get() >= 0.1

    # Assert timer is stopped
    with pytest.raises(RuntimeError):
        timer.stop()


def test_warnings_and_error():
    """Test the warnings and errors raised by the timer"""
    timer = Timer()
    with timer:
        with pytest.warns(RuntimeWarning):
            timer.get()
        with pytest.raises(RuntimeError):
            timer.start()

    # Assert timer is stopped
    with pytest.raises(RuntimeError):
        timer.stop()
