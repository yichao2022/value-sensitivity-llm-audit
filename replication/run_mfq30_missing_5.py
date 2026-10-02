#!/usr/bin/env python3
"""
Run MFQ-30 for 5 missing models to complete the 15-model set.
"""

import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT = Path(__file__).resolve().parent

def load_items():
    """Load MFQ-30 items from canonical data."""
    items = []
    with open(PROJECT / "data" / "mfq30_canonical_items.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == "GPT-4.1":  # Use as template
                items.append({
                    "item_id": row["item_id"],
                    "dimension": row["dimension"],
                    "format": row["format"],
                    "item_text": row["item_text"],
                })
    # Deduplicate
    seen = set()
    unique_items = []
    for item in items:
        if item["item_id"] not in seen:
            seen.add(item["item_id"])
            unique_items.append(item)
    return unique_items

def generate_prompt(items):
    """Generate the two-part prompt for MFQ-30."""
    part1_items = [f"{i['item_id']}: {i['item_text']}" for i in items if i["format"] == "relevance"]
    part2_items = [f"{i['item_id']}: {i['item_text']}" for i in items if i["format"] == "judgment"]
    
    return f"""You are completing a moral foundations questionnaire.

PART 1 - RELEVANCE (0=not at all relevant, 5=extremely relevant):
When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?

{chr(10).join(part1_items)}

PART 2 - AGREEMENT (0=strongly disagree, 5=strongly agree):
{chr(10).join(part2_items)}

Respond with JSON only: {{"CA01": 4, "CA02": 3, ...}}"""

def call_openai(api_key, model, prompt, base_url=None):
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=4096
    )
    return resp.choices[0].message.content

def call_anthropic(api_key, model, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=4096, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text

def parse_json(text):
    import re
    if not text:
        return None
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except:
        return None

def main():
    items = load_items()
    prompt = generate_prompt(items)
    
    # 5 missing models
    models = [
        ("Claude Haiku", "Anthropic", "claude-3-haiku-20240307", 
         lambda p: call_anthropic(os.getenv("ANTHROPIC_API_KEY"), "claude-3-haiku-20240307", p)),
        ("Claude Opus 4.8", "Anthropic", "claude-opus-4-8-20250501", 
         lambda p: call_anthropic(os.getenv("ANTHROPIC_API_KEY"), "claude-opus-4-8-20250501", p)),
        ("Llama 3.1 70B Instruct", "Meta", "meta-llama/llama-3.1-70b-instruct", 
         lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "meta-llama/llama-3.1-70b-instruct", p, "https://openrouter.ai/api/v1")),
        ("Llama 3.1 8B Instruct", "Meta", "meta-llama/llama-3.1-8b-instruct", 
         lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "meta-llama/llama-3.1-8b-instruct", p, "https://openrouter.ai/api/v1")),
        ("Mistral Large", "Mistral", "mistralai/mistral-large", 
         lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "mistralai/mistral-large", p, "https://openrouter.ai/api/v1")),
    ]
    
    results = []
    
    for name, provider, api_model, caller in models:
        print(f"\n[{name}]...", end=" ", flush=True)
        try:
            raw = caller(prompt)
            parsed = parse_json(raw)
            if parsed:
                for item_id, score in parsed.items():
                    # Find item details
                    item = next((i for i in items if i["item_id"] == item_id), None)
                    if item:
                        results.append({
                            "model": name,
                            "provider": provider,
                            "api_model": api_model,
                            "item_id": item_id,
                            "dimension": item["dimension"],
                            "format": item["format"],
                            "item_text": item["item_text"],
                            "raw_score": score,
                        })
                print(f"✓ {len(parsed)} items")
            else:
                print(f"✗ Parse failed")
                # Add empty rows for tracking
                for item in items:
                    results.append({
                        "model": name, "provider": provider, "api_model": api_model,
                        "item_id": item["item_id"], "dimension": item["dimension"],
                        "format": item["format"], "item_text": item["item_text"],
                        "raw_score": None,
                    })
        except Exception as e:
            print(f"✗ Error: {e}")
            for item in items:
                results.append({
                    "model": name, "provider": provider, "api_model": api_model,
                    "item_id": item["item_id"], "dimension": item["dimension"],
                    "format": item["format"], "item_text": item["item_text"],
                    "raw_score": None,
                })
    
    # Save results
    output_path = PROJECT / "outputs" / "mfq30_missing_5models.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "provider", "api_model", "item_id", "dimension", "format", "item_text", "raw_score"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\nDone. {len(results)} rows → {output_path}")

if __name__ == "__main__":
    main()
