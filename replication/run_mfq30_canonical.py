#!/usr/bin/env python3
"""
Canonical MFQ-30 Pipeline (Two-part 0-5 scale)
Reads data/mfq30_canonical_items.csv → 1 API call per model (32 items) → fills scores.
Supports resumption from outputs/mfq30_canonical_responses_filled.csv.

Canonical MFQ-30 design:
- Part 1 (Relevance): 16 items, 0-5, anchor: "not at all relevant" → "extremely relevant"
- Part 2 (Judgment): 16 items, 0-5, anchor: "strongly disagree" → "strongly agree"
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_canonical_items.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_responses_filled.csv"

# ── Provider configs ─────────────────────────────────────────────
PROVIDER_CONFIG = {
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "Google": {
        "env_key": "GEMINI_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "temperature": 0,
        "extra_body": {},
    },
    "Qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "qwen-max",
    },
    "Meta": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "meta-llama/llama-3.1-70b-instruct",
    },
    "Moonshot": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "moonshotai/kimi-k2.5",
    },
    "Cohere": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "cohere/command-r-plus",
    },
    "Mistral": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "mistralai/mistral-large-2",
    },
}

OPENAI_COMPATIBLE = {"OpenAI", "DeepSeek", "Qwen", "Meta", "Moonshot", "Cohere", "Mistral"}

# ── Canonical MFQ-30 Prompt ──────────────────────────────────────
def generate_mfq_prompt(items: list) -> str:
    """Generate two-part MFQ-30 prompt with correct anchors."""
    
    # Separate Part 1 and Part 2 items
    part1_items = [(i, t) for i, d, f, t in items if f == "relevance"]
    part2_items = [(i, t) for i, d, f, t in items if f == "judgment"]
    
    part1_text = "\n".join([f"{item_id}. {text}" for item_id, text in part1_items])
    part2_text = "\n".join([f"{item_id}. {text}" for item_id, text in part2_items])
    
    return f"""You will complete a moral foundations questionnaire in two parts.

=== PART 1: RELEVANCE ===
When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?

Rate each consideration from 0 (not at all relevant) to 5 (extremely relevant).

{part1_text}

=== PART 2: AGREEMENT ===
Please indicate your agreement with the following statements.

Rate each statement from 0 (strongly disagree) to 5 (strongly agree).

{part2_text}

=== INSTRUCTIONS ===
Provide your responses as a JSON object with item IDs as keys and numeric values (0-5) as values. Example: {{"CA01": 4, "CA02": 3, ...}}

Respond only with the JSON object, no additional text."""

MAX_RETRIES = 5
BASE_DELAY = 2
MAX_DELAY = 120


def load_input() -> list[dict]:
    """Read mfq30_canonical_items.csv and group items by model."""
    rows = []
    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_model(rows: list[dict]) -> list[dict]:
    """Group rows into per-model batches."""
    groups = defaultdict(lambda: {"model": "", "provider": "", "api_model": "", "items": []})
    for r in rows:
        key = r["model"]
        groups[key]["model"] = r["model"]
        groups[key]["provider"] = r["provider"]
        groups[key]["api_model"] = r["api_model"]
        groups[key]["items"].append((r["item_id"], r["dimension"], r["format"], r["item_text"]))
    return list(groups.values())


def load_existing_scores() -> dict:
    """Load already-filled scores for resumption."""
    scores = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["model"], row["item_id"])
                scores[key] = row.get("raw_score", "").strip()
    return scores


def call_api(provider: str, api_model: str, prompt: str) -> dict:
    """Make API call with retry logic."""
    import openai
    
    cfg = PROVIDER_CONFIG[provider]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing API key: {cfg['env_key']}")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=cfg.get("base_url")
    )
    
    model_to_use = cfg.get("model_override", api_model)
    
    for attempt in range(MAX_RETRIES):
        try:
            if provider in OPENAI_COMPATIBLE:
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=cfg["temperature"],
                    max_tokens=2048,
                    **cfg.get("extra_body", {})
                )
            else:
                # Native APIs (Anthropic, Google) would need separate handling
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=cfg["temperature"],
                    max_tokens=2048,
                )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                
                # Try to find JSON object
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                
                parsed = json.loads(content)
                return {"success": True, "data": parsed}
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"JSON parse error: {e}", "raw": content}
                
        except Exception as e:
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    
    return {"success": False, "error": "Max retries exceeded"}


def main():
    print("=" * 60)
    print("Canonical MFQ-30 Pipeline (Two-part 0-5 scale)")
    print("=" * 60)
    
    # Load input data
    rows = load_input()
    models = group_by_model(rows)
    existing_scores = load_existing_scores()
    
    print(f"\nTotal models to process: {len(models)}")
    print(f"Items per model: 32 (16 relevance + 16 judgment)")
    print(f"Scale: 0-5 (canonical MFQ-30)")
    
    # Process each model
    results = []
    for i, model_group in enumerate(models, 1):
        model_name = model_group["model"]
        provider = model_group["provider"]
        api_model = model_group["api_model"]
        items = model_group["items"]
        
        print(f"\n[{i}/{len(models)}] Processing {model_name}...")
        
        # Check if already completed
        all_keys = [(model_name, item_id) for item_id, _, _, _ in items]
        if all(k in existing_scores and existing_scores[k] for k in all_keys):
            print(f"  ✓ Already completed, skipping")
            for item_id, dim, fmt, text in items:
                key = (model_name, item_id)
                results.append({
                    "model": model_name,
                    "provider": provider,
                    "api_model": api_model,
                    "item_id": item_id,
                    "dimension": dim,
                    "format": fmt,
                    "item_text": text,
                    "raw_score": existing_scores[key],
                })
            continue
        
        # Generate prompt and call API
        prompt = generate_mfq_prompt(items)
        print(f"  Calling API (provider={provider}, model={api_model})...")
        
        response = call_api(provider, api_model, prompt)
        
        if response["success"]:
            scores = response["data"]
            print(f"  ✓ Got {len(scores)} scores")
            
            for item_id, dim, fmt, text in items:
                score = scores.get(item_id, "")
                results.append({
                    "model": model_name,
                    "provider": provider,
                    "api_model": api_model,
                    "item_id": item_id,
                    "dimension": dim,
                    "format": fmt,
                    "item_text": text,
                    "raw_score": str(score) if score != "" else "",
                })
        else:
            print(f"  ✗ Failed: {response.get('error', 'Unknown error')}")
            if "raw" in response:
                print(f"    Raw: {response['raw'][:200]}...")
            
            # Add empty scores for failed model
            for item_id, dim, fmt, text in items:
                results.append({
                    "model": model_name,
                    "provider": provider,
                    "api_model": api_model,
                    "item_id": item_id,
                    "dimension": dim,
                    "format": fmt,
                    "item_text": text,
                    "raw_score": "",
                })
        
        # Small delay between models
        time.sleep(1)
    
    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "provider", "api_model", "item_id", "dimension",
            "format", "item_text", "raw_score"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(results)}")
    print(f"Models: {len(models)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
