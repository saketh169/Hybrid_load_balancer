import random
from typing import List

MAX_TASK_LOAD = 100.0

class WorkloadGenerator:
    def __init__(self):
        self._current: List[float] = []

    def generate_workload(self, num_tasks: int) -> List[float]:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")
        self._current = []
        for _ in range(num_tasks):
            r = random.random()
            if r < 0.7:
                load = random.uniform(5, 25)
            elif r < 0.9:
                load = random.uniform(25, 60)
            else:
                load = random.uniform(60, MAX_TASK_LOAD)
            self._current.append(round(load, 2))
        return self._current.copy()

    def get_current_workload(self) -> List[float]:
        if not self._current:
            raise RuntimeError("No workload generated yet.")
        return self._current.copy()

    def has_workload(self) -> bool:
        return bool(self._current)

workload_generator = WorkloadGenerator()