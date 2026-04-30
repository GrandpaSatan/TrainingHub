from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkDefinition:
    id: str
    family: str
    label: str
    description: str
    smoke_default: int
    full_default: int
    worker_job_type: str
    lm_eval_tasks: tuple[str, ...]

    def public_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "label": self.label,
            "description": self.description,
            "smoke_default": self.smoke_default,
            "full_default": self.full_default,
        }


BENCHMARK_CATALOG: tuple[BenchmarkDefinition, ...] = (
    BenchmarkDefinition(
        id="gsm8k",
        family="Math",
        label="GSM8K",
        description="Grade-school arithmetic word problems with final-answer extraction.",
        smoke_default=10,
        full_default=1319,
        worker_job_type="benchmark",
        lm_eval_tasks=("gsm8k",),
    ),
    BenchmarkDefinition(
        id="math-500",
        family="Math",
        label="MATH-500",
        description="Competition-style math subset used for high-signal reasoning checks.",
        smoke_default=10,
        full_default=500,
        worker_job_type="benchmark",
        lm_eval_tasks=("math_500",),
    ),
    BenchmarkDefinition(
        id="asdiv",
        family="Math",
        label="ASDiv",
        description="Diverse arithmetic word problems for robustness across phrasing.",
        smoke_default=10,
        full_default=2305,
        worker_job_type="benchmark",
        lm_eval_tasks=("asdiv",),
    ),
    BenchmarkDefinition(
        id="gsm-symbolic",
        family="Math",
        label="GSM Symbolic",
        description="Symbol-shifted GSM-style prompts for arithmetic generalization.",
        smoke_default=10,
        full_default=500,
        worker_job_type="benchmark",
        lm_eval_tasks=("gsm_symbolic",),
    ),
    BenchmarkDefinition(
        id="sat-math",
        family="Math",
        label="SAT Math",
        description="Standardized exam math prompts for mainstream quantitative checks.",
        smoke_default=10,
        full_default=220,
        worker_job_type="benchmark",
        lm_eval_tasks=("sat_math",),
    ),
    BenchmarkDefinition(
        id="theoremqa",
        family="Math",
        label="TheoremQA",
        description="Theorem-grounded mathematical reasoning questions.",
        smoke_default=10,
        full_default=800,
        worker_job_type="benchmark",
        lm_eval_tasks=("theoremqa",),
    ),
    BenchmarkDefinition(
        id="mmlu",
        family="Knowledge",
        label="MMLU",
        description="Broad multitask knowledge and reasoning across academic subjects.",
        smoke_default=10,
        full_default=14042,
        worker_job_type="benchmark_mmlu",
        lm_eval_tasks=("mmlu",),
    ),
    BenchmarkDefinition(
        id="hellaswag",
        family="Reasoning",
        label="HellaSwag",
        description="Commonsense continuation selection for grounded reasoning quality.",
        smoke_default=10,
        full_default=10042,
        worker_job_type="benchmark_hellaswag",
        lm_eval_tasks=("hellaswag",),
    ),
    BenchmarkDefinition(
        id="arc",
        family="Reasoning",
        label="ARC",
        description="AI2 science questions covering easy and challenge splits.",
        smoke_default=10,
        full_default=7791,
        worker_job_type="benchmark_arc",
        lm_eval_tasks=("arc_challenge", "arc_easy"),
    ),
    BenchmarkDefinition(
        id="ifeval",
        family="Instruction-following",
        label="IFEval",
        description="Instruction-following checks with verifiable prompt constraints.",
        smoke_default=10,
        full_default=541,
        worker_job_type="benchmark_ifeval",
        lm_eval_tasks=("ifeval",),
    ),
    BenchmarkDefinition(
        id="humaneval",
        family="Code",
        label="HumanEval",
        description="Python function synthesis tasks for code-generation sanity checks.",
        smoke_default=5,
        full_default=164,
        worker_job_type="benchmark_code",
        lm_eval_tasks=("humaneval",),
    ),
)

BENCHMARKS_BY_ID = {definition.id: definition for definition in BENCHMARK_CATALOG}


def benchmark_catalog_payload() -> list[dict[str, Any]]:
    return [definition.public_payload() for definition in BENCHMARK_CATALOG]


def require_benchmark_definitions(benchmark_ids: list[str]) -> list[BenchmarkDefinition]:
    unknown = [benchmark_id for benchmark_id in benchmark_ids if benchmark_id not in BENCHMARKS_BY_ID]
    if unknown:
        raise ValueError(f"Unsupported benchmark id(s): {', '.join(unknown)}")
    return [BENCHMARKS_BY_ID[benchmark_id] for benchmark_id in benchmark_ids]


def benchmark_job_type_for(benchmark_ids: list[str]) -> str:
    definitions = require_benchmark_definitions(benchmark_ids)
    non_math_definition = next(
        (definition for definition in definitions if definition.worker_job_type != "benchmark"),
        None,
    )
    return non_math_definition.worker_job_type if non_math_definition else "benchmark"
