from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker, write_jsonl


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    context.check_cancelled()
    target_count = int(payload.get("target_count", 100))
    seed_prompt = payload.get("seed_prompt", "Create a concise math tutoring example.")
    teacher_model = payload.get("teacher_model", "local")
    output_path = context.job_dir / "candidate_examples.jsonl"
    report_path = context.job_dir / "validation_report.json"
    can_run_teacher = _looks_like_gguf_path(str(teacher_model)) or real_workers_enabled()
    if can_run_teacher and payload.get("use_teacher_model", False) and not payload.get("dry_run", False):
        rows = _generate_with_teacher(context, payload, target_count)
    else:
        if payload.get("use_teacher_model", False) and teacher_model != "local":
            context.event("fallback", "Teacher generation is unavailable for this target. Using deterministic local examples.", "warning")
        rows = _generate_deterministic_examples(context, seed_prompt, target_count)
    context.check_cancelled()
    rows = _deduplicate(rows)
    report = {
        "teacher_model": teacher_model,
        "requested_count": target_count,
        "candidate_count": len(rows),
        "strict": bool(payload.get("validation_strictness", "normal") == "strict"),
        "approved": False,
        "notes": "Generated examples require UI approval before downstream use.",
    }
    write_jsonl(output_path, rows)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    context.register_artifact(output_path, "generated_dataset", "Generated candidate examples", report)
    context.register_artifact(report_path, "validation_report", "Generation validation report", report)


def _generate_deterministic_examples(context: WorkerContext, seed_prompt: str, target_count: int) -> list[dict[str, Any]]:
    random.seed(int(context.job_id.split("_")[1]) if "_" in context.job_id else 7)
    categories = ["arithmetic", "algebra", "word-problem"]
    difficulties = ["easy", "medium", "hard"]
    rows = []
    for index in range(target_count):
        context.check_cancelled()
        left = random.randint(2, 40)
        right = random.randint(2, 40)
        answer = left + right
        category = categories[index % len(categories)]
        difficulty = difficulties[index % len(difficulties)]
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "You are a careful math tutor."},
                    {"role": "user", "content": f"{seed_prompt} Problem {index + 1}: What is {left} + {right}?"},
                    {"role": "assistant", "content": f"{left} + {right} = {answer}. The final answer is {answer}."},
                ],
                "metadata": {
                    "id": f"generated_{index + 1:04d}",
                    "final_answer": str(answer),
                    "category": category,
                    "difficulty": difficulty,
                    "source": "traininghub-local-generator",
                    "split": "train",
                    "tags": ["generated", category],
                    "notes": "Pending human review.",
                },
            }
        )
        context.metric({"generated": index + 1, "target_count": target_count})
    return rows


def _generate_with_teacher(context: WorkerContext, payload: dict[str, Any], target_count: int) -> list[dict[str, Any]]:
    teacher_model = str(payload["teacher_model"])
    if _looks_like_gguf_path(teacher_model):
        return _generate_with_llama_cpp(context, payload, target_count, Path(teacher_model))
    return _generate_with_transformers(context, payload, target_count)


def _generate_with_transformers(context: WorkerContext, payload: dict[str, Any], target_count: int) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("Transformers and torch are required for non-GGUF teacher generation.")

    teacher_model = payload["teacher_model"]
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    rows = []
    for index in range(target_count):
        context.check_cancelled()
        prompt = (
            f"{payload.get('seed_prompt', '')}\n"
            "Return one JSON object with fields prompt, response, final_answer, category, difficulty."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.9)),
            max_new_tokens=int(payload.get("max_tokens", 256)),
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        parsed = _parse_model_json(text)
        rows.append(_row_from_parsed(index, parsed, text, teacher_model))
        context.metric({"generated": index + 1, "target_count": target_count})
    return rows


def _generate_with_llama_cpp(
    context: WorkerContext,
    payload: dict[str, Any],
    target_count: int,
    teacher_model_path: Path,
) -> list[dict[str, Any]]:
    llama_cli = _llama_cli_path()
    if not llama_cli.exists():
        raise RuntimeError(f"llama.cpp CLI not found: {llama_cli}")
    if not teacher_model_path.exists():
        raise RuntimeError(f"GGUF teacher model not found: {teacher_model_path}")

    rows = []
    for index in range(target_count):
        context.check_cancelled()
        prompt = (
            f"{payload.get('seed_prompt', '')}\n"
            "Return exactly one JSON object with fields prompt, response, final_answer, category, difficulty. "
            "The example must be a high-quality math tutoring record."
        )
        command = [
            str(llama_cli),
            "-m",
            str(teacher_model_path),
            "-p",
            prompt,
            "-n",
            str(int(payload.get("max_tokens", 256))),
            "--temp",
            str(float(payload.get("temperature", 0.7))),
            "--top-p",
            str(float(payload.get("top_p", 0.9))),
        ]
        gpu_layers = os.getenv("LLAMA_CPP_N_GPU_LAYERS", "999")
        if gpu_layers:
            command.extend(["-ngl", gpu_layers])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=int(payload.get("generation_timeout_seconds", 300)),
        )
        parsed = _parse_model_json(result.stdout)
        rows.append(_row_from_parsed(index, parsed, result.stdout, str(teacher_model_path)))
        context.metric({"generated": index + 1, "target_count": target_count})
    return rows


def _row_from_parsed(index: int, parsed: dict[str, Any], raw_text: str, teacher_model: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a careful math tutor."},
            {"role": "user", "content": parsed.get("prompt", f"Generated problem {index + 1}")},
            {"role": "assistant", "content": parsed.get("response", raw_text[-1000:])},
        ],
        "metadata": {
            "id": f"generated_{index + 1:04d}",
            "final_answer": str(parsed.get("final_answer", "")),
            "category": parsed.get("category", "generated"),
            "difficulty": parsed.get("difficulty", "medium"),
            "source": teacher_model,
            "split": "train",
            "tags": ["generated"],
            "notes": "Pending human review.",
        },
    }


def _looks_like_gguf_path(value: str) -> bool:
    return value.casefold().endswith(".gguf")


def _llama_cli_path() -> Path:
    if os.getenv("LLAMA_CPP_CLI"):
        return Path(os.environ["LLAMA_CPP_CLI"]).expanduser()
    root = Path(os.getenv("LLAMA_CPP_ROOT", "/home/jhernandez/llama.cpp")).expanduser()
    for candidate in [root / "build" / "bin" / "llama-cli", root / "llama-cli"]:
        if candidate.exists():
            return candidate
    return root / "build" / "bin" / "llama-cli"


def _parse_model_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = json.dumps(row["messages"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


if __name__ == "__main__":
    sys.exit(run_worker(main))
