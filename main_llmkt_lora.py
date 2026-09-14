"""Submission entrypoint — `llmkt_lora`: the scored A1 model WITHOUT the 16 GB of weights.

This file is the `main.py` of the 2026-07-06 clamped-v2 LLMKT submission
(`submission_archive/2026-07-06_1255_qwen3-8b-full-final-clamped.zip`, platform smoke
**0.4546**, full test **0.6062 / AUROC 0.6332**) with **one** change: where the model
comes from.

    scored zip   assets/qwen3_8b_full_merged/   16.4 GB of merged bf16 weights
    this zip     Qwen/Qwen3-8B from the runtime mount  +  assets/lora_.../ (175 MB)

The runtime added `Qwen/Qwen3-8B` to `runtime/huggingface_models.txt`, and the runtime
**mounts** every listed model at `/code_execution/huggingface_models/<repo_id>/` (it does
NOT populate the HF cache — that assumption cost us a platform smoke on 2026-08-02, see
docs/SUBMITTING.md). `peft` 0.18.1 is in the image, so the LoRA adapter that *produced*
those merged weights can be re-applied at inference instead of shipped pre-merged.

**Reconstruction fidelity.** `scripts/merge_llmkt_lora.py` built the scored weights by
loading the base in bf16 **on CPU**, applying the adapter, and calling `merge_and_unload()`.
peft upcasts the `B @ A` product to fp32 on CPU and only there, so a GPU-side merge would
give (slightly) different bf16 weights. This file therefore repeats the offline recipe
exactly — load on CPU, merge on CPU, then `.to(device)` — which reproduces the scored
weights bit for bit (verified by `scripts/verify_lora_merge_parity.py`).

Everything downstream (prompt assembly, chat template, truncation, the True/False logit
read, the frozen Platt calibrator) is the byte-identical vendored `magnificat` package and
calibrator from that zip. Rules 3 and 4 are unchanged: every sample is scored independently
and nothing derived from the test data is logged. The `BASE_CHECK` / `WEIGHT_CHECK` lines
hash **our own** files and the mounted base — never test content.

Expected zip layout (produced by ``scripts/build_llmkt_lora_zip.py``):

    main.py
    assets/
        model_kind.txt              # "llmkt_lora"
        llmkt_calibrator.joblib     # frozen Platt, byte-identical to the scored zip
        magnificat/                 # vendored code, byte-identical to the scored zip
        lora_qwen3_8b_full_final/   # adapter_config.json + adapter_model.safetensors
        tokenizer/                  # the scored zip's tokenizer files (prompt parity)
        base_sha256.txt             # sha256 of the base shards we merged against
        weights_sha256.txt          # sha256 of the adapter file (upload-corruption guard)
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the runtime log clean (500-line cap)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(ASSETS))  # make the vendored `magnificat` importable

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submission.csv")
KIND_PATH = ASSETS / "model_kind.txt"
EPS = 1e-6  # clip probabilities so log loss never sees exactly 0 or 1

LORA_DIR = ASSETS / "lora_qwen3_8b_full_final"
TOKENIZER_DIR = ASSETS / "tokenizer"
BASE_REPO = "Qwen/Qwen3-8B"
#: Where the runtime mounts every model listed in `runtime/huggingface_models.txt`.
#: Checked in order. `LLMKT_BASE_DIR` is a local-testing escape hatch only — on the
#: platform the mount is always there, and a run that falls back proves nothing.
_HF_MOUNTS = ("/code_execution/huggingface_models", "huggingface_models")


def _resolve_base_ref():
    """(ref, mode) for the base Qwen3-8B weights.

    `mode` is "mount" (the documented runtime mechanism), "env" (local testing), or
    "cache" — the last is a bare repo id handed to transformers with
    `local_files_only=True`, which costs nothing if we are wrong about the mount and
    saves the run if the image caches models instead. It never touches the network.
    """
    import os  # noqa: PLC0415

    for mount in _HF_MOUNTS:
        cand = Path(mount) / BASE_REPO
        if (cand / "config.json").exists():
            return cand, "mount"
    env = os.environ.get("LLMKT_BASE_DIR", "")
    if env and (Path(env) / "config.json").exists():
        return Path(env), "env"
    return BASE_REPO, "cache"


def _sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest.strip()
    return out


def _load_base_plus_lora(base_ref, device: str):
    """Rebuild the scored merged model: mounted base + our adapter, merged on CPU.

    Mirrors `magnificat.llmkt_infer.load_model` (same tokenizer settings, same
    flash-attn-2 -> sdpa fallback) and `scripts/merge_llmkt_lora.py` (CPU bf16 load,
    `merge_and_unload`, then move to the GPU).
    """
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # Tokenizer ships with the submission rather than coming from the mount: it is the
    # exact file set the scored run used, so the prompts are byte-identical no matter
    # which revision of the base repo the runtime happens to mount.
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"      # last real token at attention_mask.sum()-1
    tokenizer.truncation_side = "left"    # keep the END

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_ref), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", local_files_only=True,
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            str(base_ref), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )
    model = PeftModel.from_pretrained(model, str(LORA_DIR), local_files_only=True)
    model = model.merge_and_unload()      # CPU merge == the offline merge, bit for bit
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _write_flat(submission_format, value: float) -> None:
    out = submission_format.copy()
    out["probability"] = max(EPS, min(1 - EPS, value))
    out.to_csv(SUBMISSION_PATH, index=False)


def run_llmkt_lora() -> None:
    """Qwen3-8B dialogue-KT inference from the mounted base + shipped LoRA: per
    (session, LO) read the True/False logit, apply the frozen Platt calibrator, write
    probabilities. Each sample is scored independently (rule 3); no test content is
    logged (rule 4)."""
    import os  # noqa: PLC0415
    # A1 windowing (docs/13): feed the WHOLE transcript at an 8192-token budget.
    # llmkt_infer reads LLMKT_MAX_SEQ_LEN at import time, so set it BEFORE the import.
    os.environ.setdefault("LLMKT_MAX_SEQ_LEN", "8192")
    # A local dir alone still lets transformers HEAD the hub for a newer revision --
    # a five-retry stall against blackholed DNS. Forbid it.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch  # noqa: PLC0415

    from magnificat import llmkt_infer as li  # noqa: PLC0415

    feats = pd.read_csv(DATA_DIR / "test_features.csv", dtype=str)
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    calibrator = joblib.load(ASSETS / "llmkt_calibrator.joblib")

    # 1) The adapter is the only weight file we ship. `unzip` writes bytes even when it
    #    reports a bad CRC, so safetensors would silently load garbage; re-hash against
    #    the manifest generated at pack time (our own file -- DQ-safe).
    adapter_ok = True
    expected = _read_manifest(LORA_DIR / "weights_sha256.txt")
    for name, digest in expected.items():
        ok = (LORA_DIR / name).exists() and _sha256(LORA_DIR / name) == digest
        adapter_ok = adapter_ok and ok
        print(f"WEIGHT_CHECK {name} {'OK' if ok else 'CORRUPT'}")
    if not expected:
        adapter_ok = False
        print("WEIGHT_CHECK adapter manifest MISSING")

    # 2) The base comes from the runtime mount, so we cannot guarantee its bytes -- but
    #    we can report whether they are the ones the scored model was merged against.
    #    A mismatch is not fatal (a re-uploaded revision is still the same model), it is
    #    the first thing to look at if the score moves.
    base_ref, mode = _resolve_base_ref()
    print(f"BASE_MODEL mode={mode}")
    base_match = None
    if mode in ("mount", "env"):
        base_expected = _read_manifest(ASSETS / "base_sha256.txt")
        if base_expected:
            base_match = True
            for name, digest in base_expected.items():
                p = Path(base_ref) / name
                state = "MISSING" if not p.exists() else (
                    "MATCH" if _sha256(p) == digest else "DIFFERENT")
                base_match = base_match and state == "MATCH"
                print(f"BASE_CHECK {name} {state}")
    print(f"BASE_WEIGHTS={'AS_SCORED' if base_match else 'UNVERIFIED'}")

    if not adapter_ok:
        print("MODEL_INTEGRITY=CORRUPT the upload corrupted the adapter; predictions are "
              "INVALID, please RE-UPLOAD (this score is not real)")
        _write_flat(submission_format, float(calibrator.transform([0.5])[0]))
        return
    print("MODEL_INTEGRITY=OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Never die without saying where: on any failure write a valid base-rate submission
    # and print the exception TYPE and location only -- never its message, which could
    # quote a path or, in principle, test content (rule 4). Same crash-proofing as the
    # 2026-07-01 self-diagnosing build.
    try:
        model, tokenizer = _load_base_plus_lora(base_ref, device=device)
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"MODEL_LOAD_FAILED {type(exc).__name__} at "
              f"{Path(tb.filename).name}:{tb.lineno} (base mode={mode}); predictions are "
              "INVALID, please RE-UPLOAD (this score is not real)")
        _write_flat(submission_format, float(calibrator.transform([0.5])[0]))
        return
    true_token, false_token = li.get_true_false_tokens(tokenizer)

    # ---- from here down: verbatim from the scored zip's main.py -------------------
    # assemble each session's dialogue once (sessions are shared across responses)
    dia_cache: dict[str, object] = {}
    prompts, rids = [], []
    fallback_rids = []
    for _, r in feats.iterrows():
        sid = r["session_id"]
        if sid not in dia_cache:
            tx_path = DATA_DIR / "test_transcripts" / f"{sid}.csv"
            base = li.build_session_dialogue(pd.read_csv(tx_path)) if tx_path.exists() else None
            dia_cache[sid] = base
        base = dia_cache[sid]
        if not base:
            fallback_rids.append(r["response_id"])
            continue
        # A1: full transcript (max_turns=0 keeps every turn); llmkt_infer then
        # left-truncates to the 8192-token budget set above.
        dialogue = li.build_terminal_dialogue(base, max_turns=0)
        prompts.append(li.build_prompt(tokenizer, dialogue, str(r["learning_objective"]).strip()))
        rids.append(r["response_id"])

    raw = li.score_prompts(model, tokenizer, prompts, true_token, false_token,
                           batch_size=8, device=device)
    cal = calibrator.transform(raw)
    proba = pd.Series(cal, index=rids)
    # responses with no usable transcript fall back to the calibrated base rate
    base_rate = float(calibrator.transform([0.5])[0]) if fallback_rids else 0.5
    for rid in fallback_rids:
        proba.loc[rid] = base_rate

    out = submission_format.copy()
    out["probability"] = out["response_id"].map(proba).fillna(base_rate).clip(EPS, 1 - EPS)
    out.to_csv(SUBMISSION_PATH, index=False)
    print("inference complete")


def main() -> None:
    kind = KIND_PATH.read_text().strip() if KIND_PATH.exists() else "llmkt_lora"
    if kind != "llmkt_lora":
        raise SystemExit(f"this main.py only serves kind 'llmkt_lora' (got {kind!r})")
    run_llmkt_lora()


if __name__ == "__main__":
    main()
