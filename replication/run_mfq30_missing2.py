#!/usr/bin/env python3
"""补跑缺失的 2 个模型: Claude Opus 4 + Gemini 2.5 Flash (thinking)"""
import csv, json, os, re, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_canonical_items.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_final.csv"

MISSING_MODELS = ["Claude Opus 4", "Gemini 2.5 Flash (thinking)"]

# 正确的模型 ID 映射
MODEL_ID_MAP = {
    "Claude Opus 4": "claude-opus-4-8",  # 不是 claude-opus-4
    "Gemini 2.5 Flash (thinking)": "gemini-2.5-flash-thinking-exp-03-25"  # 实验版本
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

def call_anthropic(api_model, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # Opus 不支持 temperature 参数
    kwargs = {"model": api_model, "max_tokens": 4096,
              "system": "You are a helpful assistant. Output only valid JSON.",
              "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = client.messages.create(temperature=0, **kwargs)
    except Exception:
        resp = client.messages.create(**kwargs)
    return resp.content[0].text

def call_gemini(api_model, prompt):
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=api_model,
        contents=prompt,
        config={"temperature": 0, "max_output_tokens": 4096}
    )
    return resp.text

def parse_json(content):
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        content = json_match.group(0)
    return json.loads(content)

def main():
    # Load items for missing models
    rows = []
    with open(INPUT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] in MISSING_MODELS:
                rows.append(row)

    # Group by model
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["model"]].append(r)

    results = []
    for model_name in MISSING_MODELS:
        model_rows = groups[model_name]
        items = [(r["item_id"], r["dimension"], r["format"], r["item_text"]) for r in model_rows]
        provider = model_rows[0]["provider"]
        api_model = MODEL_ID_MAP.get(model_name, model_rows[0]["api_model"])  # 使用正确的模型 ID
        prompt = generate_mfq_prompt(items)

        print(f"\n[{model_name}] provider={provider}, api_model={api_model}")
        for attempt in range(3):
            try:
                if provider == "Anthropic":
                    content = call_anthropic(api_model, prompt)
                elif provider == "Google":
                    content = call_gemini(api_model, prompt)
                else:
                    raise RuntimeError(f"Unknown provider: {provider}")

                scores = parse_json(content)
                print(f"  ✓ Got {len(scores)} scores")

                for r in model_rows:
                    score = scores.get(r["item_id"], "")
                    results.append({
                        "model": model_name, "provider": provider,
                        "api_model": api_model, "item_id": r["item_id"],
                        "dimension": r["dimension"], "format": r["format"],
                        "item_text": r["item_text"], "raw_score": str(score) if score != "" else ""
                    })
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  ✗ Failed after 3 attempts")

    # Append to canonical_final.csv
    if results:
        # Read existing
        existing = []
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                existing.append(row)

        # Append new
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing:
                writer.writerow(row)
            for row in results:
                writer.writerow(row)

        print(f"\n✅ Appended {len(results)} rows to {OUTPUT_CSV}")
    else:
        print("\n✗ No results to append")

if __name__ == "__main__":
    main()
