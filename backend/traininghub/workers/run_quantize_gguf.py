from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker


SUPPORTED_QUANT_TYPES = {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    source_gguf = Path(payload["source_gguf"])
    quant_type = payload.get("quant_type", "Q4_K_M")
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise RuntimeError(f"Unsupported quantization type: {quant_type}")
    if any(marker in source_gguf.name.upper() for marker in SUPPORTED_QUANT_TYPES):
        raise RuntimeError("Refusing to re-quantize an already quantized GGUF.")
    output_path = context.job_dir / f"{source_gguf.stem}.{quant_type}.gguf"
    llama_cpp_root = Path(payload.get("llama_cpp_root") or os.getenv("LLAMA_CPP_ROOT", "/home/jhernandez/llama.cpp"))
    quantize_binary = Path(payload.get("quantize_binary") or llama_cpp_root / "build" / "bin" / "llama-quantize")
    command = [str(quantize_binary), str(source_gguf), str(output_path), quant_type]
    metadata = {
        "source_gguf": str(source_gguf),
        "output_path": str(output_path),
        "quant_type": quant_type,
        "command": command,
    }
    context.write_metadata("quantize_gguf_command.json", metadata)
    if real_workers_enabled() and not payload.get("dry_run", False):
        if not quantize_binary.exists():
            raise RuntimeError(f"llama-quantize not found: {quantize_binary}")
        context.run_command(command)
    else:
        output_path.write_text(f"smoke quantized gguf placeholder {quant_type}\n", encoding="utf-8")
    context.register_artifact(output_path, "gguf_quantized", f"Quantized GGUF {quant_type}", metadata)


if __name__ == "__main__":
    sys.exit(run_worker(main))
