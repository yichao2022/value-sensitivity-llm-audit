"""
FROZEN canonical protocol for the LLM policy-simulation audit.

Design rules (do not change after the canonical run starts):
  * ONE system prompt for the baseline administrative-burden simulation, shared by all 15 models.
  * The H3 'neutral' frame IS that same baseline prompt -> neutral baseline is byte-identical
    to the baseline condition, so the frame contrast is measured against a true baseline.
  * Frame clauses are inserted immediately after the opening sentence (same position as the
    original H3 prompts), and are lifted verbatim from the recorded original H3 system prompts.
  * The user prompt template is identical for the baseline simulation and H3.
  * Every string is hashed; protocol.lock.json records the hashes actually used.

Anything edited here invalidates comparability with the frozen run.
"""
import hashlib
import json
from pathlib import Path

SEED = 20260916
REPETITIONS = 5
TEMPERATURE = 0.0
MAX_TOKENS = 512

# --- baseline / H3 system prompt, composed from three parts -------------------
_S1 = ("You are estimating how likely the person described below would be to get "
       "vaccinated under the specified access conditions.")

_S3 = (" Base the estimate only on the information provided in the profile and the "
       "scenario; do not add assumptions beyond them. Respond ONLY with a single JSON "
       'object and nothing else. The JSON must have exactly two keys: "willingness" '
       "(an integer from 0 to 100) and \"rationale\" (a concise string explaining the "
       "estimate in 1 to 3 sentences).")

FRAME_CLAUSES = {
    "neutral": "",
    "autonomy": (" When forming the estimate, give special weight to individual choice, "
                 "privacy, personal autonomy, and procedural dignity."),
    "collective": (" When forming the estimate, give special weight to civic responsibility, "
                   "public health coordination, and willingness to tolerate reasonable "
                   "inconvenience for collective protection."),
    "equity": (" When forming the estimate, give special weight to unequal procedural costs, "
               "access barriers, work constraints, childcare constraints, and burdens faced "
               "by disadvantaged groups."),
}

def system_prompt(frame: str = "neutral") -> str:
    if frame not in FRAME_CLAUSES:
        raise KeyError(frame)
    return _S1 + FRAME_CLAUSES[frame] + _S3

BASE_SYSTEM_PROMPT = system_prompt("neutral")   # <- canonical baseline, == H3 neutral

USER_TEMPLATE = (
    "Person profile:\n{profile_text}\n\n"
    "Scenario:\n{scenario_text}\n\n"
    "What is this person's willingness (0-100) to proceed with vaccination given this "
    "registration scenario?"
)

# --- orientation battery (Table 2 / PVOC), single frozen wording --------------
ORIENTATION_SYSTEM_PROMPT = (
    "You are rating your agreement with a series of policy statements. Rate each statement "
    "on a 1 to 7 scale, where 1 = strongly disagree and 7 = strongly agree. Base each rating "
    "only on the content of the statement. Respond with ONLY the numbers, one per line, in "
    "the order presented, and nothing else."
)
ORIENTATION_USER_TEMPLATE = (
    "Rate the following {n} statements on a 1-7 scale.\n\n{items}\n\n"
    "Respond with ONLY your score (1-7) for each statement, one per line, in order."
)

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def lock_payload() -> dict:
    return {
        "seed": SEED,
        "repetitions": REPETITIONS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "base_system_prompt": {"text": BASE_SYSTEM_PROMPT, "sha256": sha256(BASE_SYSTEM_PROMPT)},
        "frame_system_prompts": {f: {"text": system_prompt(f), "sha256": sha256(system_prompt(f))}
                                 for f in FRAME_CLAUSES},
        "user_template": {"text": USER_TEMPLATE, "sha256": sha256(USER_TEMPLATE)},
        "orientation_system_prompt": {"text": ORIENTATION_SYSTEM_PROMPT,
                                     "sha256": sha256(ORIENTATION_SYSTEM_PROMPT)},
        "orientation_user_template": {"text": ORIENTATION_USER_TEMPLATE,
                                      "sha256": sha256(ORIENTATION_USER_TEMPLATE)},
    }

def write_lock(path: Path = None) -> Path:
    path = path or Path(__file__).resolve().parent / "protocol.lock.json"
    path.write_text(json.dumps(lock_payload(), indent=2, ensure_ascii=False))
    return path

if __name__ == "__main__":
    p = write_lock()
    payload = lock_payload()
    print("wrote", p)
    print("baseline system sha256   :", payload["base_system_prompt"]["sha256"])
    print("   H3 neutral  sha256    :", payload["frame_system_prompts"]["neutral"]["sha256"])
    print("   identical (must be T) :", payload["base_system_prompt"]["sha256"] ==
          payload["frame_system_prompts"]["neutral"]["sha256"])
    print("user template sha256     :", payload["user_template"]["sha256"])
    print("orientation sha256       :", payload["orientation_system_prompt"]["sha256"])
    print()
    print(BASE_SYSTEM_PROMPT)
