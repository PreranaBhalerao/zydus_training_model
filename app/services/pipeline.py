"""
Invoice extraction fallback pipeline.

Logic copied verbatim from validation.py sections 4–12
(helpers + generate + run_pipeline). Do not alter behavior.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy


# ============================================================
# MODEL CONFIGURATION (from validation.py)
# ============================================================

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
# NORMALIZATION HELPERS
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
# JSON PARSING
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
# IRN EXTRACTION
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
# REGEX FALLBACK
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
# HEADER FALLBACK
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
# MODEL GENERATION
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
# INFERENCE PIPELINE
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
