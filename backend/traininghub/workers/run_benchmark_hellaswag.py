from __future__ import annotations

import sys
from typing import Any

from traininghub.workers._lm_eval_runner import run_benchmark_worker
from traininghub.workers.common import WorkerContext, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    run_benchmark_worker(context, payload, "HellaSwag benchmark results")


if __name__ == "__main__":
    sys.exit(run_worker(main))
