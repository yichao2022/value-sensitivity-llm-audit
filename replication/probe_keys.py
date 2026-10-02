#!/usr/bin/env python3
"""Probe all 6 providers with a minimal 1-token request to check key liveness.
Usage: python3 probe_keys.py [--models]  (--models: probe all 15 model ids instead of 6 providers)
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "OpenAI":   {"env": "OPENAI_API_KEY",   "kind": "openai",  "model": "gpt-4o-mini"},
    "Anthropic": {"env": "ANTHROPIC_API_KEY", "kind": "anthropic", "model": "claude-sonnet-4-6"},
    "Google":   {"env": "GEMINI_API_KEY",   "kind": "google",  "model": "gemini-2.5-flash"},
    "DeepSeek": {"env": "DEEPSEEK_API_KEY", "kind": "openai",  "model": "deepseek-chat",
                 "base_url": "https://api.deepseek.com"},
    "Qwen":     {"env": "DASHSCOPE_API_KEY", "kind": "openai",  "model": "qwen-max",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "Meta/OpenRouter": {"env": "OPENROUTER_API_KEY", "kind": "openai", "model": "meta-llama/llama-3.1-8b-instruct",
                 "base_url": "https://openrouter.ai/api/v1"},
}

def probe_openai(cfg):
    from openai import OpenAI
    key = os.environ.get(cfg["env"])
    if not key:
        return "MISSING KEY"
    client = OpenAI(api_key=key, **({"base_url": cfg["base_url"]} if cfg.get("base_url") else {}))
    resp = client.chat.completions.create(
        model=cfg["model"], max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the number 1."}],
        temperature=0)
    return f"OK ({resp.choices[0].message.content.strip()!r})"

def probe_anthropic(cfg):
    from anthropic import Anthropic
    key = os.environ.get(cfg["env"])
    if not key:
        return "MISSING KEY"
    client = Anthropic(api_key=key)
    resp = client.messages.create(
        model=cfg["model"], max_tokens=5,
        system="Reply with the number 1.",
        messages=[{"role": "user", "content": "Hi"}])
    return f"OK ({resp.content[0].text.strip()!r})"

def probe_google(cfg):
    from google import genai
    key = os.environ.get(cfg["env"])
    if not key:
        return "MISSING KEY"
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=cfg["model"], contents="Reply with the number 1.")
    return f"OK ({resp.text.strip()!r})"

def main():
    only_models = "--models" in sys.argv
    print(f"Probing {'15 model ids' if only_models else '6 providers'}...\n")
    for name, cfg in PROVIDERS.items():
        try:
            if cfg["kind"] == "openai":
                r = probe_openai(cfg)
            elif cfg["kind"] == "anthropic":
                r = probe_anthropic(cfg)
            else:
                r = probe_google(cfg)
            print(f"  [{'OK ' if r.startswith('OK') else 'FAIL'}] {name:<18} {r}")
        except Exception as e:
            msg = str(e)
            # classify common errors
            low = msg.lower()
            if "401" in low or "invalid" in low or "unauthorized" in low or "api key" in low:
                cls = "AUTH FAIL"
            elif "402" in low or "insufficient" in low or "balance" in low or "quota" in low or "billing" in low:
                cls = "NO CREDIT"
            elif "429" in low or "rate" in low:
                cls = "RATE LIMITED"
            else:
                cls = "ERROR"
            print(f"  [FAIL] {name:<18} {cls}: {msg[:150]}")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
