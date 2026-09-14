"""Prompts and schemas for the docs/29 interview-derived labels.

Provenance: [`docs/29_interviews_and_labels.md`](../../docs/29_interviews_and_labels.md)
— three semi-structured interviews with practising maths tutors. Every code below
traces to a numbered interview; the mapping is in the docstring of each block.

**Scope is set by the 2026-08-04 prevalence gate**, not by docs/29 as written.
`scripts/doc29_prevalence_probe.py` over all 22,821 train sessions killed two of the
proposed codes outright:

    NEGATIVE_SELF_EFFICACY   0.9% of sessions  ->  dropped
    CURIOSITY_EXTENSION      3.1% of sessions  ->  dropped
    DISENGAGEMENT            defined by absence, not lexically probeable -> timestamps

so the affect pass here is **two** codes, not the five of docs/29 §7.4.

Two design choices worth knowing:

1. **Separate label spaces, not extra codes on the NTO taxonomy.** docs/29 §4.1 warns
   that adding codes to the saturated 16-code per-utterance instrument "buys little and
   dilutes an already-saturated label space". Running these as their own passes means the
   NTO block is untouched and any effect is attributable.

2. **`NONE` is explicit, never an omission.** F3's distilled tutor head under-predicts
   coverage (0.508 true -> 0.363 predicted) precisely because the annotator's silence and
   its "no code applies" were the same event. Here every utterance gets a record, so the
   distillation target includes a real negative class and the rate denominators are exact.
   Same reasoning as docs/29 §7.4's "absence must be recorded, not merely omitted".
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# STUDENT PASS — three orthogonal fields per student utterance.
#
#   reasoning   docs/29 §5.3  — split of DEMONSTRATING_UNDERSTANDING.
#                               Interview 2: "not memorising formulas",
#                               "I would teach them to think about the problem".
#   confidence  docs/29 §7.3  — epistemic stance. The document's own strongest
#                               hypothesis; crosses with the validated per-turn
#                               correctness annotator to give calibration cells.
#   affect      docs/29 §7.4  — Interview 3: "way more enthusiastic ... huge relief".
#                               Cut to 2 codes by the prevalence gate.
#
# One pass, three labels: the generation cost is paid once and the fields are
# deliberately orthogonal — docs/29 §7.4 notes that forcing one label per turn
# "destroys the interaction that makes it interesting".
# ---------------------------------------------------------------------------

DOC29_REASONING = ["NOT_ANSWERING", "ANSWER_ONLY", "REASONING_SHOWN"]
DOC29_CONFIDENCE = ["na", "hedged", "unmarked", "assertive"]
DOC29_AFFECT = ["NONE", "INSIGHT_DELIGHT", "FRUSTRATION"]

DOC29_STUDENT_FIELDS = {
    "reasoning": DOC29_REASONING,
    "confidence": DOC29_CONFIDENCE,
    "affect": DOC29_AFFECT,
}

DOC29_STUDENT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "utterance_id": {"type": "integer"},
            "reasoning": {"type": "string", "enum": DOC29_REASONING},
            "confidence": {"type": "string", "enum": DOC29_CONFIDENCE},
            "affect": {"type": "string", "enum": DOC29_AFFECT},
        },
        "required": ["utterance_id", "reasoning", "confidence", "affect"],
        "additionalProperties": False,
    },
}

DOC29_STUDENT_PROMPT = """\
You are an expert educational discourse analyst. For each STUDENT utterance in a 1:1 \
maths tutoring session, label three INDEPENDENT things. Return ONLY a JSON array — no \
outer object, no explanation.

These are ASR transcripts of spoken sessions, so expect disfluency, [unclear] markers and \
short backchannels. Judge what the student is doing, not how polished it sounds.

## Field 1 — reasoning: is the student answering, and do they show their thinking?
- NOT_ANSWERING: the turn is not an attempt at the mathematics — backchannel ("yeah", \
  "mm-hmm"), a question to the tutor, social or logistical talk, reading the problem aloud, \
  or an unintelligible turn.
- ANSWER_ONLY: a substantive answer or claim with NO visible thinking. A bare number or \
  statement. "48." / "It's a half." / "The third one."
- REASONING_SHOWN: working, a justification, or a stated method — anything that reveals HOW \
  they got there. "48, because 6 times 8." / "You flip the second fraction then multiply." / \
  "I did the brackets first."
Judge visible reasoning only. A correct bare answer is still ANSWER_ONLY; a wrong answer \
with working shown is still REASONING_SHOWN.

## Field 2 — confidence: the student's stance on their OWN answer
Only meaningful when the student is asserting something.
- na: reasoning is NOT_ANSWERING, so there is no claim to be confident about.
- hedged: explicit uncertainty markers, or the answer delivered as a question. \
  "I think it's 12?" / "Maybe a quarter" / "Is it 8?" / "12... I guess"
- assertive: stated flatly as fact, or with an explicit confidence marker. \
  "It's 12." / "Definitely a quarter." / "That's just 8."
- unmarked: an assertion with neither uncertainty nor emphasis. This is the DEFAULT and \
  should be the most common value.
Stance is independent of correctness — a confidently wrong answer is assertive.

## Field 3 — affect: a marked emotional event, if and only if one occurs
Most turns are affectively flat. Use NONE unless the marker is clear in the text.
- NONE: no marked affect. THIS IS THE COMMON CASE.
- INSIGHT_DELIGHT: the moment of getting it — "ohh!", "wait, I get it now", "so THAT's why", \
  audible relief after a stretch of difficulty. Requires a genuine realisation, not a polite \
  "oh okay" acknowledgement of something the tutor said.
- FRUSTRATION: distress directed at the problem or the task — "I can't do this", "this is \
  impossible", "ugh", giving up mid-attempt. Not mere difficulty; the student must express \
  the feeling.

Output format, one object per utterance, ALL utterances included:
[{"utterance_id": <int>, "reasoning": "<...>", "confidence": "<...>", "affect": "<...>"}, ...]"""


# ---------------------------------------------------------------------------
# TUTOR PASS — one docs/29 move per tutor utterance, plus explicit NONE.
#
#   NORMALIZING_DIFFICULTY     §5.4  Interview 1 "removing the fear",
#                                    Interview 3 "comfortable not knowing"
#   RELEASE_HANDOFF            §5.1  Interviews 2 and 3, independently:
#                                    "first do something together, then the student alone"
#   TRANSFER_PROBE             §5.2  Interview 2's own test of real learning
#   SIMPLIFYING_TO_SUBPROBLEM  §6    Interview 1 "simplify, get to the inner function"
#   MOTIVATING_RELEVANCE       §6    Interview 1 "say what it is for"
#   SUMMARIZING_PROGRESS       §6    Interview 2's recipe for a stuck student
#   OFFERING_CHOICE            §6    Interview 2 "let the student pick the problem"
#
# These are moves the NTO taxonomy has no word for (docs/29 §3). Where one could
# be confused with an NTO code, the prompt says so explicitly — the distinctions
# ARE the contribution, so they are spelled out rather than assumed.
# ---------------------------------------------------------------------------

DOC29_TUTOR_CODES = [
    "NORMALIZING_DIFFICULTY",
    "RELEASE_HANDOFF",
    "TRANSFER_PROBE",
    "SIMPLIFYING_TO_SUBPROBLEM",
    "MOTIVATING_RELEVANCE",
    "SUMMARIZING_PROGRESS",
    "OFFERING_CHOICE",
    "NONE",
]

DOC29_TUTOR_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "utterance_id": {"type": "integer"},
            "code": {"type": "string", "enum": DOC29_TUTOR_CODES},
        },
        "required": ["utterance_id", "code"],
        "additionalProperties": False,
    },
}

DOC29_TUTOR_PROMPT = """\
You are an expert educational discourse analyst. For each TUTOR utterance in a 1:1 maths \
tutoring session, decide whether it performs one of seven specific pedagogical moves. \
Return ONLY a JSON array — no outer object, no explanation.

These are ASR transcripts of spoken sessions. Most tutor turns will be NONE — the seven \
moves below are specific things, and ordinary explaining, hinting, questioning and \
feedback are NOT among them. Use NONE freely.

Allowed codes (exactly one per utterance):

- NORMALIZING_DIFFICULTY: explicitly licenses not-knowing or being stuck. Makes the \
  difficulty a property of the problem rather than the student. "That's a really tricky \
  one." / "It's fine to be stuck here." / "Loads of people get that wrong first time." \
  NOT praise (which is about performance), and NOT a bare "don't worry" attached to \
  logistics.

- RELEASE_HANDOFF: hands the current problem over for the student to attempt alone, after \
  the tutor has been doing or leading it. "Now you try this one." / "Your turn." / "See if \
  you can do the next one yourself." The defining feature is the transfer of who is \
  working, not the question mark.

- TRANSFER_PROBE: offers a DIFFERENT problem of about the SAME difficulty testing the same \
  idea, to check whether the understanding holds. "Same method, but with 3/8 this time." / \
  "What about if it was a hexagon instead?" NOT a repeat of the identical problem, and NOT \
  a harder follow-on or the next item in a worksheet.

- SIMPLIFYING_TO_SUBPROBLEM: replaces the current problem with a smaller or easier version \
  — friendlier numbers, one step of it, a stripped-down case. "Forget the fractions for a \
  second — what's 4 times 3?" / "Let's just do the bracket first." NOT a hint about the \
  existing problem; the tutor must change what is being worked on.

- MOTIVATING_RELEVANCE: states where the idea is used or why it matters. "This is how shops \
  work out the sale price." / "You'll need this for the area questions." NOT an illustrative \
  example or analogy used to explain the mechanics — this is about purpose, not illustration.

- SUMMARIZING_PROGRESS: aggregates what has been established so far across several turns. \
  "So we've got the numerator, and we said the denominator was 8, and now we need..." NOT a \
  near-verbatim restatement of the student's last turn (that is a different move), and NOT a \
  summary of what the tutor alone just explained.

- OFFERING_CHOICE: hands the student control over WHAT to work on. "Which one do you want to \
  try?" / "Shall we do the algebra or the geometry first?" NOT a rhetorical "shall we?" that \
  moves on regardless, and NOT offering a choice of method within a fixed problem.

- NONE: anything else, including explaining, hinting, asking a guiding question, giving \
  feedback, giving the answer, and all logistics and social talk. Most turns are NONE.

Output format, one object per utterance, ALL utterances included:
[{"utterance_id": <int>, "code": "<CODE>"}, ...]"""


#: passes this module defines, in the shape `annotate_doc29_labels.py` consumes
DOC29_PASSES = {
    "doc29_student": {
        "role": "student",
        "prompt": DOC29_STUDENT_PROMPT,
        "schema": DOC29_STUDENT_SCHEMA,
        "fields": DOC29_STUDENT_FIELDS,
        "batch_size": 25,
        "max_tokens": 2048,
        "max_prompt_tokens": 5000,
    },
    "doc29_tutor": {
        "role": "tutor",
        "prompt": DOC29_TUTOR_PROMPT,
        "schema": DOC29_TUTOR_SCHEMA,
        "fields": {"code": DOC29_TUTOR_CODES},
        "batch_size": 40,
        "max_tokens": 1200,
        "max_prompt_tokens": 6000,
    },
}
