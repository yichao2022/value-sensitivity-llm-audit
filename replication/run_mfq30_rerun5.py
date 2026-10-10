#!/usr/bin/env python3
"""补跑 4 个空分数模型 + 修复 Command R+"""
import csv, json, os, re, time, sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_canonical_items.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_final.csv"

NEED_RERUN = ["GPT-4.1 nano", "GPT-4o", "Llama 3.1 70B Instruct", "Mistral Large 2", "Command R+"]

PROVIDER_CONFIG = {
    "OpenAI": {"env_key": "OPENAI_API_KEY", "base_url": None, "temperature": 0},
    "Meta": {"env_key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
             "model_override": "meta-llama/llama-3.1-70b-instruct"},
    "Mistral": {"env_key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
                "model_override": "mistralai/mistral-large-2407"},
    "Cohere": {"env_key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
               "model_override": "cohere/command-r-plus"},
}

def generate_mfq_prompt(items):
    part1 = [(i, t) for i, d, f, t in items if f == "relevance"]
    part2 = [(i, t) for i, d, f, t in items if f == "judgment"]
    p1 = "\n".join([f"{i}. {t}" for i, t in part1])
    p2 = "\n".join([f"{i}. {t}" for i, t in part2])
    return f"""You will complete a moral foundations questionnaire in two parts.

=== PART 1: RELEVANCE ===
When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?

Rate each consideration from 0 (not at all relevant) to 5 (extremely relevant).

{p1}

=== PART 2: AGREEMENT ===
Please indicate your agreement with the following statements.

Rate each statement from 0 (strongly disagree) to 5 (strongly agree).

{p2}

=== INSTRUCTIONS ===
Provide your responses as a JSON object with item IDs as keys and numeric values (0-5) as values. Example: {{"CA01": 4, "CA02": 3, ...}}

Respond only with the JSON object, no additional text."""

def call_openai_compatible(provider, api_model, prompt):
    import openai
    cfg = PROVIDER_CONFIG[provider]
    api_key = os.getenv(cfg["env_key"])
    client = openai.OpenAI(api_key=api_key, base_url=cfg.get("base_url"))
    model = cfg.get("model_override", api_model)
    
    # OpenRouter doesn't support temperature=0
    kwargs = {"model": model, "messages": [
        {"role": "system", "content": "You are a helpful assistant. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ], "max_tokens": 4096}
    if provider not in ["Meta", "Mistral", "Cohere"]:  # These use OpenRouter
        kwargs["temperature"] = cfg["temperature"]
    
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()

def parse_json(content):
    m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if m: content = m.group(1)
    m = re.search(r'\{[\s\S]*\}', content)
    if m: content = m.group(0)
    return json.loads(content)

def main():
    # Load items
    rows = []
    with open(INPUT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] in NEED_RERUN:
                rows.append(row)

    groups = defaultdict(list)
    for r in rows:
        groups[r["model"]].append(r)

    new_scores = {}  # (model, item_id) -> score

    for model_name in NEED_RERUN:
        model_rows = groups[model_name]
        items = [(r["item_id"], r["dimension"], r["format"], r["item_text"]) for r in model_rows]
        provider = model_rows[0]["provider"]
        api_model = model_rows[0]["api_model"]
        prompt = generate_mfq_prompt(items)

        print(f"\n[{model_name}] provider={provider}, api_model={api_model}")
        for attempt in range(3):
            try:
                content = call_openai_compatible(provider, api_model, prompt)
                scores = parse_json(content)
                print(f"  ✓ Got {len(scores)} scores")
                for item_id, _, _, _ in items:
                    new_scores[(model_name, item_id)] = str(scores.get(item_id, ""))
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  ✗ Failed after 3 attempts")

    # Update canonical_final.csv
    if new_scores:
        existing = []
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                key = (row["model"], row["item_id"])
                if key in new_scores:
                    row["raw_score"] = new_scores[key]
                existing.append(row)

        # Deduplicate: keep last occurrence per (model, item_id)
        seen = {}
        for row in existing:
            key = (row["model"], row["item_id"])
            seen[key] = row

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in seen.values():
                writer.writerow(row)

        print(f"\n✅ Updated {len(new_scores)} scores in {OUTPUT_CSV}")
        print(f"✅ Deduplicated to {len(seen)} rows")
    else:
        print("\n✗ No scores to update")

if __name__ == "__main__":
    main()
