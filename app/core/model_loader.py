"""
Load base Qwen model and both LoRA adapters once at startup.

Adapter loading mirrors validation.py sections 14–17.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from app.core.config import BASE_MODEL, MAX_SEQ, V1_PATH, V3_PATH

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    loaded_adapters: List[str]


def _count_adapter_keys(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(
        str(path / "adapter_model.safetensors"),
        framework="pt",
    ) as f:
        return len(list(f.keys()))


def _present_keys(model, adapter_name: str):
    return [
        k
        for k in model.state_dict()
        if (
            f"lora_A.{adapter_name}.weight" in k
            or f"lora_B.{adapter_name}.weight" in k
        )
    ]


def load_adapters_present(model, adapters):
    """Same behavior as validation.load_adapters_present."""
    from peft import PeftModel

    loaded = []
    first = True

    for name, path in adapters:
        path = Path(path)

        assert (path / "adapter_config.json").exists(), (
            f"MISSING folder/config: {path}"
        )

        assert (path / "adapter_model.safetensors").exists(), (
            f"MISSING weights file: {path}"
        )

        n_file = _count_adapter_keys(path)

        if first:
            model = PeftModel.from_pretrained(
                model,
                str(path),
                adapter_name=name,
            )
            first = False
        else:
            if (
                hasattr(model, "peft_config")
                and name in model.peft_config
            ):
                try:
                    model.delete_adapter(name)
                except Exception:
                    pass

            model.load_adapter(
                str(path),
                adapter_name=name,
            )

        keys = _present_keys(model, name)

        assert name in model.peft_config, (
            f"MISSING adapter in peft_config: {name}"
        )

        assert len(keys) == n_file, (
            f"MISSING weights for '{name}': "
            f"model has {len(keys)} tensors, "
            f"file has {n_file}"
        )

        logger.info(
            "PRESENT %s: %s/%s tensors",
            name,
            len(keys),
            n_file,
        )

        loaded.append(name)

    model.set_adapter(loaded[0])
    return model, loaded


def _assert_runtime_ready() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available. CUDA is required to load the "
            "Qwen + LoRA invoice models."
        )

    for label, path in (("v3", V3_PATH), ("v1", V1_PATH)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} adapter folder not found: {path}"
            )
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Missing {label} adapter_config.json: {path}"
            )
        if not (path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                f"Missing {label} adapter_model.safetensors: {path}. "
                "Place the LoRA weight files next to adapter_config.json "
                "before starting the API."
            )


def load_model_bundle() -> ModelBundle:
    """Load base model and both adapters; fail fast on missing assets."""
    _assert_runtime_ready()

    from unsloth import FastLanguageModel

    logger.info("Loading base model: %s", BASE_MODEL)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=True,
    )

    model, loaded_names = load_adapters_present(
        model,
        [
            ("v3", V3_PATH),
            ("v1", V1_PATH),
        ],
    )

    FastLanguageModel.for_inference(model)

    logger.info(
        "Adapters loaded OK (all PRESENT): %s",
        ", ".join(loaded_names),
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        loaded_adapters=loaded_names,
    )
