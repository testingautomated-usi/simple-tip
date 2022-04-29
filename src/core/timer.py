""" A utility class to time operations."""
import time
import warnings


class Timer:
    """A utility class to time operations."""

    def __init__(self, start=False):
        self._start_time = None
        self._elapsed = 0
        if start:
            self.start()

    def start(self):
        """Starts the timer. Timer must not be started yet."""
        if self._start_time is not None:
            raise RuntimeError("Timer is already started")

        self._start_time = time.time()

    def stop(self):
        """Stops the timer. Timer must be started and not yet stopped."""
        if self._start_time is None:
            raise RuntimeError("Timer is not started")

        self._elapsed += time.time() - self._start_time
        self._start_time = None

    def timed(self, f):
        """Decorator to time a passed function."""
        # docstr-coverage:excused `explained in outer function`
        def wrapper(*args, **kwargs):
            with self:
                return f(*args, **kwargs)

        return wrapper

    def get(self):
        """Returns the elapsed time in seconds."""
        if self._start_time is not None:
            warnings.warn("Timer is not stopped", RuntimeWarning)

        return self._elapsed

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
