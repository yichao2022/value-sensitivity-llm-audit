"""
Headroom SDK integration — compress messages before API calls.

Works with ALL providers (DeepSeek, Qwen, Meta, OpenAI, Anthropic, Google).
No proxy needed — uses headroom.compress() directly.

Set HEADROOM_ENABLED=1 to activate. Unset = zero overhead.
"""

import os

ENABLED = os.environ.get("HEADROOM_ENABLED", "0") == "1"

_compress = None
_stats = {"calls": 0, "tokens_before": 0, "tokens_after": 0}

def _get_compress():
    global _compress
    if _compress is None:
        from headroom import compress as _c
        _compress = _c
    return _compress


def compress_messages(messages: list[dict], model: str = "gpt-4") -> list[dict]:
    """Compress a message list through headroom. Returns original if disabled."""
    if not ENABLED:
        return messages
    try:
        result = _get_compress()(messages, model=model)
        _stats["calls"] += 1
        _stats["tokens_before"] += result.tokens_before
        _stats["tokens_after"] += result.tokens_after
        return result.messages
    except Exception:
        return messages  # fail-open: use originals


def stats() -> dict:
    """Return compression stats."""
    saved = _stats["tokens_before"] - _stats["tokens_after"]
    ratio = saved / _stats["tokens_before"] if _stats["tokens_before"] else 0
    return {**_stats, "tokens_saved": saved, "compression_ratio": f"{ratio:.1%}"}


def print_stats():
    """Print compression summary."""
    s = stats()
    if s["calls"] == 0:
        print("[headroom] No calls compressed.")
        return
    print(f"[headroom] {s['calls']} calls | "
          f"{s['tokens_before']} → {s['tokens_after']} tokens | "
          f"saved {s['tokens_saved']} ({s['compression_ratio']})")


if ENABLED:
    print("[headroom] ✓ compression enabled")
