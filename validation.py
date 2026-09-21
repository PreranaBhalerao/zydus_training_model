# ============================================================
# ZYDUS QWEN INVOICE MODEL VALIDATION
# ============================================================
# Runs validation for:
#   - invoice_lora_v3
#   - invoice_lora
#
# Base model:
#   unsloth/Qwen2.5-1.5B-Instruct
#
# No training is performed.
#
# Expected repository structure:
#
# zydus_training_model/
# ├── invoice_lora/
# │   ├── adapter_config.json
# │   ├── adapter_model.safetensors
# │   └── ...
# ├── invoice_lora_v3/
# │   ├── adapter_config.json
# │   ├── adapter_model.safetensors
# │   └── ...
# └── validation/
#     └── validate_models.py
#
# Validation dataset:
# Set VAL_PATH environment variable in Colab/local environment.
#
# Example:
# os.environ["VAL_PATH"] = "/content/drive/MyDrive/training_export/training_export/pdfplumber_val.jsonl"
# ============================================================


from pathlib import Path
import os
import json
import re
import gc
from copy import deepcopy


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Find repository root.
#
# If this script is located at:
#
# zydus_training_model/
# └── validation/
#     └── validate_models.py
#
# then:
# Path(__file__).resolve().parent.parent
# points to:
#
# zydus_training_model/
#
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Useful when running directly inside a Colab notebook/cell.
    REPO_ROOT = Path.cwd()


# ------------------------------------------------------------
# Trained LoRA adapters
# ------------------------------------------------------------

V3_PATH = REPO_ROOT / "invoice_lora_v3"
V1_PATH = REPO_ROOT / "invoice_lora"


# ------------------------------------------------------------
# Validation dataset
# ------------------------------------------------------------
#
# IMPORTANT:
# Do NOT hard-code your personal Google Drive path.
#
# Set:
#
# os.environ["VAL_PATH"] = "/your/path/pdfplumber_val.jsonl"
#
# before running the validation.
#
VAL_PATH_ENV = os.environ.get("VAL_PATH")


if VAL_PATH_ENV:
    VAL_PATH = Path(VAL_PATH_ENV).expanduser().resolve()
else:
    # Optional fallback if validation data is stored inside repo.
    VAL_PATH = REPO_ROOT / "validation" / "pdfplumber_val.jsonl"


print("=" * 88)
print("PROJECT PATHS")
print("=" * 88)
print("REPO_ROOT =", REPO_ROOT)
print("V3_PATH   =", V3_PATH)
print("V1_PATH   =", V1_PATH)
print("VAL_PATH  =", VAL_PATH)
print("=" * 88)


# ============================================================
# 2. PATH VALIDATION
# ============================================================

assert V3_PATH.exists(), (
    f"V3 adapter folder not found:\n{V3_PATH}\n\n"
    "Make sure the GitHub repository contains invoice_lora_v3/"
)

assert V1_PATH.exists(), (
    f"V1 adapter folder not found:\n{V1_PATH}\n\n"
    "Make sure the GitHub repository contains invoice_lora/"
)

assert (V3_PATH / "adapter_config.json").exists(), (
    f"Missing V3 adapter_config.json:\n{V3_PATH}"
)

assert (V3_PATH / "adapter_model.safetensors").exists(), (
    f"Missing V3 adapter_model.safetensors:\n{V3_PATH}"
)

assert (V1_PATH / "adapter_config.json").exists(), (
    f"Missing V1 adapter_config.json:\n{V1_PATH}"
)

assert (V1_PATH / "adapter_model.safetensors").exists(), (
    f"Missing V1 adapter_model.safetensors:\n{V1_PATH}"
)

assert VAL_PATH.exists(), (
    f"Validation dataset not found:\n{VAL_PATH}\n\n"
    "Set VAL_PATH to the location of pdfplumber_val.jsonl."
)


# ============================================================
# 3. GPU + MODEL IMPORTS
# ============================================================

import torch

assert torch.cuda.is_available(), (
    "GPU is not available. "
    "Enable GPU: Runtime → Change runtime type → GPU"
)

from unsloth import FastLanguageModel


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

INSTRUCTION = (
    "Extract this invoice from the OCR text. "
    "Reply with JSON only, with keys invoice_summary and line_items."
)

HEADER_FIELDS = [
    "vendor",
    "vendor_gstin",
    "tax",
    "irn",
]

SCORE_FIELDS = [
    "invoice_no",
    "invoice_date",
    "vendor",
    "customer",
    "vendor_gstin",
    "customer_gstin",
    "tax",
    "total",
    "irn",
]

STRICT_FIELDS = [
    "invoice_no",
    "vendor_gstin",
    "customer_gstin",
    "tax",
    "total",
]

BLANK = {
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "-",
    "na",
    "nil",
}

GSTIN_RE = re.compile(
    r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]"
)

HEX = set("0123456789abcdefABCDEF")


# ============================================================
# 5. VALIDATION SETTINGS
# ============================================================

LIMIT = None
# None = full validation set
#
# Example for testing:
# LIMIT = 20

MAX_PROMPT_TOKENS = 6000
MAX_SEQ = 8192


# ============================================================
# 6. NORMALIZATION HELPERS
# ============================================================

def is_blank(v):
    if v is None:
        return True

    if isinstance(v, str) and v.strip().lower() in BLANK | {""}:
        return True

    return False


def norm_gstin(v):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(v or "").upper()
    )


def valid_gstin(v):
    g = norm_gstin(v)
    return bool(GSTIN_RE.fullmatch(g)), g


def norm_field(name, v):

    if is_blank(v):
        return ""

    s = str(v).strip()

    if name in {"vendor_gstin", "customer_gstin"}:
        return norm_gstin(s)

    if name in {"tax", "total"}:
        return re.sub(r"[, ]", "", s)

    if name == "irn":
        return "".join(
            ch.lower()
            for ch in s
            if ch in HEX
        )

    return re.sub(r"\s+", " ", s.upper())


# ============================================================
# 7. JSON PARSING
# ============================================================

def parse_json(text):

    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    try:
        return json.loads(text)

    except Exception:
        return None


# ============================================================
# 8. IRN EXTRACTION
# ============================================================

def extract_irn_64(ocr_text):

    text = ocr_text or ""

    m = re.search(
        r"[a-fA-F0-9]{64}",
        text
    )

    if m:
        return m.group().lower()

    gapped = re.search(
        r"[a-fA-F0-9](?:[\s\-]*[a-fA-F0-9]){63}",
        text
    )

    if gapped:
        return "".join(
            ch.lower()
            for ch in gapped.group()
            if ch in HEX
        )

    label = re.search(
        r"\bIRN(?:\s*(?:NO\.?|NUMBER|#))?\s*[:.,\-]?\s*",
        text,
        re.I
    )

    if not label:
        return None

    chars = []

    for ch in text[
        label.end():
        label.end() + 240
    ]:

        if ch in HEX:
            chars.append(ch.lower())

            if len(chars) == 64:
                return "".join(chars)

        elif ch.isspace() or ch in "-:.,":
            continue

        else:
            break

    return None


# ============================================================
# 9. REGEX FALLBACK
# ============================================================

def regex_fill(summary, ocr_text, used):

    summary = dict(summary or {})
    found = []

    for m in GSTIN_RE.finditer(
        re.sub(
            r"[^A-Z0-9]",
            "",
            (ocr_text or "").upper()
        )
    ):

        g = m.group()

        if g not in found:
            found.append(g)

    ok, _ = valid_gstin(
        summary.get("vendor_gstin")
    )

    if (not ok) and found:
        summary["vendor_gstin"] = found[0]
        used.append("regex:vendor_gstin")

    ok_c, _ = valid_gstin(
        summary.get("customer_gstin")
    )

    if (not ok_c) and len(found) >= 2:

        summary["customer_gstin"] = found[1]
        used.append("regex:customer_gstin")

    elif (
        (not ok_c)
        and len(found) == 1
        and found[0] != norm_gstin(
            summary.get("vendor_gstin")
        )
    ):

        summary["customer_gstin"] = found[0]
        used.append("regex:customer_gstin")

    if is_blank(summary.get("irn")):

        irn = extract_irn_64(ocr_text)

        if irn:
            summary["irn"] = irn
            used.append("regex:irn")

    return summary


# ============================================================
# 10. HEADER FALLBACK
# ============================================================

def needs_header_fallback(summary):

    summary = summary or {}

    for k in HEADER_FIELDS:

        if k == "vendor_gstin":

            ok, _ = valid_gstin(
                summary.get(k)
            )

            if not ok:
                return True

        elif is_blank(summary.get(k)):
            return True

    return False


def merge_header_from_v1(
    base_obj,
    v1_obj,
    used
):

    out = (
        deepcopy(base_obj)
        if isinstance(base_obj, dict)
        else {
            "invoice_summary": {},
            "line_items": None,
        }
    )

    s = out.setdefault(
        "invoice_summary",
        {}
    )

    s1 = (
        (v1_obj or {}).get("invoice_summary") or {}
        if isinstance(v1_obj, dict)
        else {}
    )

    for k in HEADER_FIELDS:

        if k == "vendor_gstin":

            ok, _ = valid_gstin(s.get(k))
            ok1, g1 = valid_gstin(s1.get(k))

            if (not ok) and ok1:
                s[k] = g1
                used.append(f"v1:{k}")

        else:

            if (
                is_blank(s.get(k))
                and not is_blank(s1.get(k))
            ):
                s[k] = s1[k]
                used.append(f"v1:{k}")

    out["invoice_summary"] = s

    return out


# ============================================================
# 11. MODEL GENERATION
# ============================================================

def generate(
    model,
    tokenizer,
    ocr_text,
    max_new_tokens=2048
):

    prompt = (
        INSTRUCTION
        + "\n\nOCR text:\n"
        + ocr_text.strip()
    )

    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    raw = tokenizer.decode(
        out[0][
            inputs["input_ids"].shape[-1]:
        ],
        skip_special_tokens=True,
    )

    return raw, parse_json(raw)


# ============================================================
# 12. INFERENCE PIPELINE
# ============================================================

def run_pipeline(
    model,
    tokenizer,
    ocr_text
):

    used = []

    # --------------------------------------------------------
    # V3 first
    # --------------------------------------------------------

    model.set_adapter("v3")

    raw_v3, obj_v3 = generate(
        model,
        tokenizer,
        ocr_text
    )

    if not isinstance(obj_v3, dict):

        obj_v3 = {
            "invoice_summary": {},
            "line_items": None,
            "_raw": raw_v3,
        }

    summary = regex_fill(
        dict(
            obj_v3.get(
                "invoice_summary"
            ) or {}
        ),
        ocr_text,
        used,
    )

    irn = extract_irn_64(ocr_text)

    if irn:

        if summary.get("irn") != irn:
            used.append("script:irn")

        summary["irn"] = irn

    obj_v3["invoice_summary"] = summary

    # --------------------------------------------------------
    # V1 fallback
    # --------------------------------------------------------

    if needs_header_fallback(summary):

        model.set_adapter("v1")

        _, obj_v1 = generate(
            model,
            tokenizer,
            ocr_text
        )

        obj_v3 = merge_header_from_v1(
            obj_v3,
            obj_v1,
            used,
        )

        s2 = regex_fill(
            dict(
                obj_v3.get(
                    "invoice_summary"
                ) or {}
            ),
            ocr_text,
            used,
        )

        if irn:
            s2["irn"] = irn

        obj_v3["invoice_summary"] = s2

    return obj_v3, used, raw_v3


# ============================================================
# 13. LINE ITEM COUNT
# ============================================================

def line_count(obj):

    if not isinstance(obj, dict):
        return None

    li = obj.get("line_items")

    if (
        isinstance(li, dict)
        and isinstance(li.get("items"), list)
    ):
        return len(li["items"])

    if isinstance(li, list):
        return len(li)

    return None


# ============================================================
# 14. LOAD TRAINED ADAPTERS
# ============================================================

from peft import PeftModel
from safetensors import safe_open


def _count_adapter_keys(path):

    with safe_open(
        str(
            Path(path)
            / "adapter_model.safetensors"
        ),
        framework="pt",
    ) as f:

        return len(list(f.keys()))


def _present_keys(
    model,
    adapter_name
):

    return [
        k
        for k in model.state_dict()
        if (
            f"lora_A.{adapter_name}.weight" in k
            or
            f"lora_B.{adapter_name}.weight" in k
        )
    ]


def load_adapters_present(
    model,
    adapters
):

    loaded = []
    first = True

    for name, path in adapters:

        path = Path(path)

        assert (
            path / "adapter_config.json"
        ).exists(), (
            f"MISSING folder/config: {path}"
        )

        assert (
            path / "adapter_model.safetensors"
        ).exists(), (
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

        keys = _present_keys(
            model,
            name
        )

        assert (
            name in model.peft_config
        ), (
            f"MISSING adapter in peft_config: {name}"
        )

        assert len(keys) == n_file, (
            f"MISSING weights for '{name}': "
            f"model has {len(keys)} tensors, "
            f"file has {n_file}"
        )

        print(
            f"PRESENT {name}: "
            f"{len(keys)}/{n_file} tensors"
        )

        loaded.append(name)

    model.set_adapter(loaded[0])

    return model, loaded


# ============================================================
# 15. CLEAN GPU MEMORY
# ============================================================

for name in (
    "model",
    "tokenizer"
):

    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 16. LOAD BASE QWEN MODEL
# ============================================================

print("\nLoading base model:")
print(BASE_MODEL)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)


# ============================================================
# 17. LOAD BOTH TRAINED ADAPTERS
# ============================================================

model, loaded_names = load_adapters_present(
    model,
    [
        ("v3", V3_PATH),
        ("v1", V1_PATH),
    ],
)

FastLanguageModel.for_inference(model)

print(
    "loaded OK (all PRESENT):",
    ", ".join(loaded_names)
)


# ============================================================
# 18. LOAD VALIDATION DATA
# ============================================================

rows = []

with VAL_PATH.open(
    encoding="utf-8"
) as f:

    for line in f:

        if (
            LIMIT is not None
            and len(rows) >= LIMIT
        ):
            break

        rec = json.loads(line)

        if rec.get("ocr_text"):
            rows.append(rec)


print(
    f"scoring {len(rows)} invoices "
    f"(LIMIT={LIMIT})"
)


# ============================================================
# 19. INITIALIZE METRICS
# ============================================================

stats = {
    k: {
        "n": 0,
        "ok": 0,
        "missing": 0,
        "wrong": 0,
    }
    for k in SCORE_FIELDS + ["line_count"]
}

strict_ok = 0
parse_fail = 0
skipped_long = 0
invoice_fail = 0


# ============================================================
# 20. RUN VALIDATION
# ============================================================

for i, row in enumerate(rows, 1):

    ocr = (
        row.get("ocr_text") or ""
    ).strip()

    prompt = (
        INSTRUCTION
        + "\n\nOCR text:\n"
        + ocr
    )

    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    ntok = len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )

    if ntok > MAX_PROMPT_TOKENS:

        skipped_long += 1
        continue

    pred, used, raw = run_pipeline(
        model,
        tokenizer,
        ocr
    )

    if not isinstance(pred, dict):

        parse_fail += 1

        pred = {
            "invoice_summary": {},
            "line_items": None,
        }

    ps = dict(
        pred.get(
            "invoice_summary"
        ) or {}
    )

    gold = (
        row.get("gemini_json")
        or {}
    )

    gs = (
        gold.get(
            "invoice_summary"
        )
        or {}
    )

    field_fail = False

    # --------------------------------------------------------
    # Field-level scoring
    # --------------------------------------------------------

    for field in SCORE_FIELDS:

        g = norm_field(
            field,
            gs.get(field)
        )

        p = norm_field(
            field,
            ps.get(field)
        )

        stats[field]["n"] += 1

        if is_blank(ps.get(field)):

            stats[field]["missing"] += 1
            field_fail = True

        elif p == g:

            stats[field]["ok"] += 1

        else:

            stats[field]["wrong"] += 1
            field_fail = True


    # --------------------------------------------------------
    # Line-item count
    # --------------------------------------------------------

    gc_ = line_count(gold)
    pc_ = line_count(pred)

    stats["line_count"]["n"] += 1

    if pc_ is None:

        stats["line_count"]["missing"] += 1
        field_fail = True

    elif pc_ == gc_:

        stats["line_count"]["ok"] += 1

    else:

        stats["line_count"]["wrong"] += 1
        field_fail = True


    # --------------------------------------------------------
    # Strict-field pass
    # --------------------------------------------------------

    if all(
        (
            not is_blank(
                ps.get(k)
            )
        )
        and
        (
            norm_field(
                k,
                ps.get(k)
            )
            ==
            norm_field(
                k,
                gs.get(k)
            )
        )
        for k in STRICT_FIELDS
    ):

        strict_ok += 1


    if field_fail:
        invoice_fail += 1


    if i % 5 == 0:

        print(
            f"scored {i}/{len(rows)}"
        )


# ============================================================
# 21. FINAL RESULTS
# ============================================================

scored = stats["invoice_no"]["n"]

print("\n" + "=" * 88)

print(
    f"{'field':<18} "
    f"{'n':>6} "
    f"{'ok':>6} "
    f"{'missing':>8} "
    f"{'wrong':>7} "
    f"{'accuracy':>10} "
    f"{'fail_rate':>10}"
)

print("-" * 88)

total_ok = 0
total_n = 0
total_fail = 0

for field, s in stats.items():

    n = s["n"] or 1

    acc = (
        100 * s["ok"] / n
    )

    fail = (
        100
        * (
            s["missing"]
            + s["wrong"]
        )
        / n
    )

    total_ok += s["ok"]
    total_n += s["n"]

    total_fail += (
        s["missing"]
        + s["wrong"]
    )

    print(
        f"{field:<18} "
        f"{s['n']:>6} "
        f"{s['ok']:>6} "
        f"{s['missing']:>8} "
        f"{s['wrong']:>7} "
        f"{acc:>9.2f}% "
        f"{fail:>9.2f}%"
    )


print("-" * 88)

overall_accuracy = (
    100 * total_ok / total_n
    if total_n
    else 0
)

overall_fail = (
    100 * total_fail / total_n
    if total_n
    else 0
)

print(
    f"{'ALL FIELDS':<18} "
    f"{total_n:>6} "
    f"{total_ok:>6} "
    f"{'':>8} "
    f"{'':>7} "
    f"{overall_accuracy:>9.2f}% "
    f"{overall_fail:>9.2f}%"
)

print("=" * 88)

print(
    f"invoices scored:     {scored}"
)

print(
    f"skipped (too long):  {skipped_long}"
)

print(
    f"parse fail:          {parse_fail}"
)

print(
    f"strict-field pass:   "
    f"{strict_ok}/{scored}  "
    f"({100 * strict_ok / scored if scored else 0:.2f}%)"
)

print(
    f"invoice fail rate:   "
    f"{invoice_fail}/{scored}  "
    f"({100 * invoice_fail / scored if scored else 0:.2f}%)"
)

print(
    "TIP: set LIMIT = None for full val set."
)
