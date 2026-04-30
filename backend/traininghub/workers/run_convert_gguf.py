from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    source_checkpoint = Path(payload["source_checkpoint"])
    output_path = context.job_dir / f"{payload.get('output_name', 'model')}.fp16.gguf"
    llama_cpp_root = Path(payload.get("llama_cpp_root") or os.getenv("LLAMA_CPP_ROOT", "/home/jhernandez/llama.cpp"))
    convert_script = llama_cpp_root / "convert_hf_to_gguf.py"
    command = [
        sys.executable,
        str(convert_script),
        str(source_checkpoint),
        "--outfile",
        str(output_path),
        "--outtype",
        payload.get("outtype", "f16"),
    ]
    context.write_metadata(
        "convert_gguf_command.json",
        {"source_checkpoint": str(source_checkpoint), "output_path": str(output_path), "command": command},
    )
    if real_workers_enabled() and not payload.get("dry_run", False):
        if not convert_script.exists():
            raise RuntimeError(f"llama.cpp conversion script not found: {convert_script}")
        context.run_command(command)
    else:
        output_path.write_text("smoke gguf placeholder\n", encoding="utf-8")
    context.register_artifact(output_path, "gguf_fp16", "FP16/BF16 GGUF checkpoint", {"command": command})


if __name__ == "__main__":
    sys.exit(run_worker(main))
