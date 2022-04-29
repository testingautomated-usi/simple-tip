"""There appears to be a memory leak in newer uwiz/tensorflow versions. This prevents it."""

import gc

from uncertainty_wizard.models.ensemble_utils import DynamicGpuGrowthContextManager


class SingleUseContext(DynamicGpuGrowthContextManager):
    """Makes sure processes are not reused for multiple models, to avoid memory leaks."""

    # docstr-coverage: inherited
    @classmethod
    def max_sequential_tasks_per_process(cls) -> int:
        return 1

    # docstr-coverage: inherited
    def __exit__(self, type, value, traceback) -> None:
        """
        Will be executed before session the model was executed. You can use this for clean up tasks.
        :return: None
        """
        super().__exit__(type, value, traceback)
        gc.collect()
