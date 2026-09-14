"""Self-contained LLMKT inference for the Trace-the-Ace submission runtime.

This module reproduces the dialogue-kt (LLMKT) inference path *exactly* but with
**zero** dependency on the `dialogue_kt` package, `peft`, or `datasets` — only
`torch` + `transformers` (both present in the official runtime). The LoRA adapter
is merged into the base weights ahead of time (see ``scripts/merge_llmkt_lora.py``),
so at inference we load a plain ``Qwen2ForCausalLM`` and read the True/False logit.

Parity contract (verified offline against the harness in ``scripts/llmkt_parity_check.py``):
the prompt strings produced here are byte-identical to what
``dialogue_kt.kt_data_loading.LMKTDatasetUnpacked`` builds for the same
(transcript, learning_objective) sample. The functions below are copied verbatim
from the harness:
  - ``add_content`` / ``process_dialogue``  <- dialogue_kt/data_loading.py
  - ``get_dialogue_text`` / ``kt_system_prompt`` / ``kt_user_prompt`` /
    ``get_true_false_tokens`` and the prompt constants  <- dialogue_kt/prompting.py
  - the terminal-turn assembly + role mapping  <- build_traceace_kt.py
  - left-truncation to 4096 + last-real-token logit read  <- kt_data_loading.py / training.py

Per the competition rules this module logs **nothing** derived from test content
and scores every sample independently (no cross-sample pooling).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch

# ---- prompt constants (verbatim from dialogue_kt/prompting.py) ----------------

KT_SYSTEM_PROMPT = """You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} Your job is to predict if the student has a particular knowledge component at the current point in the dialogue. Please follow these instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this knowledge component or not.
- Only respond with a single word, "True" or "False"."""

TRACEACE_DIALOGUE_DESC = "the student is learning about math concepts."

# Bound bf16 activation/logit memory: left-truncate so the terminal turn + LO +
# the True/False generation-prompt position (all at the END) are preserved.
# MUST match training (dialogue_kt/kt_data_loading.py); both read LLMKT_MAX_SEQ_LEN.
# Raise to 8192 for the --select full window (see docs/12).
import os as _os
MAX_SEQ_LEN = int(_os.environ.get("LLMKT_MAX_SEQ_LEN", "4096"))


# ---- dialogue assembly (verbatim from data_loading.py / build_traceace_kt.py) -

def add_content(cur: str, new: str) -> str:
    new = new.strip()
    if not cur:
        return new
    if cur == new:
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new


def process_dialogue(turns: List[dict]) -> List[dict]:
    cur_role = turns[0]["role"]
    cur_turn = {"turn": 0 if cur_role == "student" else 1, "teacher": "", "student": ""}
    result = []
    for turn in turns:
        if turn["role"] == "teacher" and cur_role == "student":
            result.append(cur_turn)
            cur_turn = {"turn": cur_turn["turn"] + 1, "teacher": "", "student": ""}
        cur_role = turn["role"]
        cur_turn[cur_role] = add_content(cur_turn[cur_role], turn["content"])
    if cur_turn["student"]:
        result.append(cur_turn)
    return result


def build_session_dialogue(tx: pd.DataFrame) -> Optional[List[dict]]:
    """One session transcript -> LLMKT turn list. background dropped."""
    tx = tx.sort_values("utterance_id")
    utts = []
    for _, u in tx.iterrows():
        role = str(u["role"]).strip().lower()
        content = "" if pd.isna(u["content"]) else str(u["content"]).strip()
        if not content:
            continue
        if role == "tutor":
            utts.append({"role": "teacher", "content": content})
        elif role == "student":
            utts.append({"role": "student", "content": content})
        # background -> dropped
    if not utts:
        return None
    return process_dialogue(utts)


def build_terminal_dialogue(base: List[dict], max_turns: int = 40) -> List[dict]:
    """Replicate build_traceace_kt: end-of-session window + synthetic terminal turn."""
    real = [dict(t) for t in base]
    if max_turns and len(real) > max_turns:
        real = real[-max_turns:]
    for i, t in enumerate(real, start=1):
        t["turn"] = i
    term = len(real) + 1
    return real + [{"turn": term, "teacher": "", "student": ""}]


# ---- prompt text (verbatim from prompting.py) ---------------------------------

def get_dialogue_text(dialogue: List[dict], turn_idx: int = None,
                      tag_wrapper: bool = True) -> str:
    lines = []
    for turn in dialogue:
        if turn["teacher"]:
            lines.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
        if turn_idx is not None and turn_idx == turn["turn"]:
            break
        if turn["student"]:
            lines.append(f"Student Turn {turn['turn']}: {turn['student']}")
    prompt = "\n".join(lines)
    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]"
    return prompt


def kt_system_prompt() -> str:
    return KT_SYSTEM_PROMPT.format(desc=TRACEACE_DIALOGUE_DESC)


def kt_user_prompt(dialogue: List[dict], turn_idx: int, kc: Optional[str]) -> str:
    prompt = get_dialogue_text(dialogue, turn_idx=turn_idx)
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def build_prompt(tokenizer, dialogue: List[dict], lo_text: str) -> str:
    """Full chat-templated prompt for the terminal turn, ending at the model's
    own generation prompt so the next token is True/False (template-agnostic)."""
    term = dialogue[-1]["turn"]
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": kt_system_prompt()},
            {"role": "user", "content": kt_user_prompt(dialogue, term, lo_text)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3: suppress the <think> block so the next token is True/False (matches
        # training, see kt_data_loading.py). Harmless for Qwen2.5/Llama templates.
        enable_thinking=False,
    )


def get_true_false_tokens(tokenizer):
    true = tokenizer("True").input_ids[-1]
    false = tokenizer("False").input_ids[-1]
    return true, false


_HDR_LEN = None


def _header_len(tokenizer) -> int:
    """Tokens of [system prompt + '<|im_start|>user\n'] under the live chat template.
    LLMKT_KEEP_HEADER=1: the plain left cut at MAX_SEQ_LEN deletes this header (the task
    instruction) on the ~15% longest transcripts; keep it and drop the OLDEST dialogue
    instead (-0.0052 LL on touched rows, sign-consistent over four LO-disjoint folds,
    figures/data/keephdr_verdict.txt). Mirrors dialogue_kt/kt_data_loading.py exactly."""
    global _HDR_LEN
    if _HDR_LEN is None:
        s = tokenizer.apply_chat_template(
            [{"role": "system", "content": kt_system_prompt()},
             {"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        cut = s.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        _HDR_LEN = len(tokenizer(s[:cut]).input_ids)
        print(f"PROBE keephdr H={_HDR_LEN} keep_tail={MAX_SEQ_LEN - _HDR_LEN}")
    return _HDR_LEN


def _encode_keep_header(tokenizer, prompts):
    H = _header_len(tokenizer)
    out = []
    for ids in tokenizer(prompts).input_ids:
        out.append(ids if len(ids) <= MAX_SEQ_LEN else ids[:H] + ids[-(MAX_SEQ_LEN - H):])
    return out


# ---- model + scoring ----------------------------------------------------------

def load_model(merged_dir: str, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"      # last real token at attention_mask.sum()-1
    tokenizer.truncation_side = "left"    # keep the END
    # Prefer flash-attn 2 if available; fall back to SDPA (always present in torch).
    # Attention impl does not change the logits, only speed/memory.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=dtype, attn_implementation="flash_attention_2",
        ).to(device)
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir, torch_dtype=dtype, attn_implementation="sdpa",
        ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_prompts(model, tokenizer, prompts: List[str], true_token: int,
                  false_token: int, batch_size: int = 8, device: str = "cuda") -> List[float]:
    """P(True) = softmax([logit_True, logit_False])[0] at the last real token.

    For traceace each terminal turn carries exactly one KC (the LO), so the
    harness's mean-ar aggregation over KCs is a no-op (single value).

    We deliberately do NOT call ``model(...)`` (the CausalLM head), because that
    materializes the full ``[batch, seq_len, vocab]`` logit tensor and casts it to
    fp32 — ~18 GB for batch 8 x seq 4096 x 152k vocab, which OOMs a 46 GB card.
    Instead we run the base model for hidden states and apply ``lm_head`` to only
    the **last real token** of each sequence. The result is numerically identical
    (same matmul on the same hidden vector) but ~vocab*seq cheaper in memory, so
    the tokenization-parity guarantee is unaffected."""
    base = model.model                       # Qwen2Model (no LM head)
    lm_head = model.get_output_embeddings()  # the tied/untied lm_head Linear

    # Length-sorted batching: group similar-length prompts so each batch pads to
    # its own max (not the global 4096), slashing wasted compute on a transcript
    # set with a wide length spread. Reordering is pure compute scheduling — each
    # sample's logit still depends only on its own tokens (rule-3 safe) — and we
    # scatter results back to the original order before returning.
    _keep_hdr = _os.environ.get("LLMKT_KEEP_HEADER") == "1"
    if _keep_hdr:
        lengths = [len(x) for x in _encode_keep_header(tokenizer, prompts)]
    else:
        lengths = [len(tokenizer(p, truncation=True, max_length=MAX_SEQ_LEN).input_ids)
                   for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    probs: List[float] = [0.0] * len(prompts)
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        chunk = [prompts[i] for i in idxs]
        if _keep_hdr:
            enc = tokenizer.pad({"input_ids": _encode_keep_header(tokenizer, chunk)},
                                return_tensors="pt").to(device)
        else:
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_SEQ_LEN,
            ).to(device)
        hidden = base(input_ids=enc.input_ids,
                      attention_mask=enc.attention_mask).last_hidden_state
        last_idxs = enc.attention_mask.sum(dim=-1) - 1
        bs = hidden.shape[0]
        last_hidden = hidden[torch.arange(bs), last_idxs]   # [bs, hidden]
        logits = lm_head(last_hidden)                       # [bs, vocab] only
        tf = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
        p_true = torch.softmax(tf.float(), dim=1)[:, 0]
        for j, i in enumerate(idxs):
            probs[i] = p_true[j].item()
    return probs
