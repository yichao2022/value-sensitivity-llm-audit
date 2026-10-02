#!/usr/bin/env python3
"""
Fix failed models with better JSON parsing.
"""

import csv
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_canonical_items.csv"

def load_items():
    items = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items

def generate_prompt(items):
    part1 = [(r['item_id'], r['item_text']) for r in items if r['format'] == 'relevance']
    part2 = [(r['item_id'], r['item_text']) for r in items if r['format'] == 'judgment']
    part1_text = "\n".join([f"{i}. {t}" for i, t in part1])
    part2_text = "\n".join([f"{i}. {t}" for i, t in part2])
    return f"""Complete this moral foundations questionnaire.

PART 1 - RELEVANCE (0=not at all relevant, 5=extremely relevant):
{part1_text}

PART 2 - AGREEMENT (0=strongly disagree, 5=strongly agree):
{part2_text}

IMPORTANT: Respond with ONLY a JSON object. Example: {{\"CA01\": 4, "CA02\": 3, ...}}
Do not include markdown formatting, explanations, or any other text."""

def call_api(api_key, model, prompt, base_url=None):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a questionnaire completion system. Output only valid JSON with no markdown or explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"}  # Force JSON mode
    )
    return resp.choices[0].message.content

def parse_json_robust(text):
    """More robust JSON parsing."""
    import re
    if not text:
        return None
    
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to extract JSON from markdown
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*\})',
        r'(\{.*\})',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except:
                pass
    
    # Try line-by-line
    result = {}
    for line in text.split('\n'):
        m = re.match(r'["\']?([A-Z]{2}\d{2})["\']?\s*[:=]\s*(\d+)', line.strip())
        if m:
            result[m.group(1)] = int(m.group(2))
    
    return result if len(result) >= 10 else None

def run_model(name, model_id, api_key, base_url=None):
    items = load_items()
    prompt = generate_prompt(items)
    
    print(f"[{name}]...", end=" ", flush=True)
    try:
        raw = call_api(api_key, model_id, prompt, base_url)
        scores = parse_json_robust(raw)
        
        if scores and len(scores) >= 10:
            print(f"✓ {len(scores)} scores")
            return {item['item_id']: scores.get(item['item_id'], '') for item in items}
        else:
            print(f"✗ Parse failed (got {len(scores) if scores else 0} scores)")
            print(f"  Raw: {raw[:200] if raw else 'None'}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    results = {}
    
    # GPT-4o
    scores = run_model("GPT-4o", "gpt-4o", os.getenv("OPENAI_API_KEY"))
    if scores:
        results["GPT-4o"] = scores
    time.sleep(1)
    
    # GPT-4.1 nano
    scores = run_model("GPT-4.1 nano", "gpt-4.1-nano", os.getenv("OPENAI_API_KEY"))
    if scores:
        results["GPT-4.1 nano"] = scores
    time.sleep(1)
    
    # Llama 3.1 70B via OpenRouter
    scores = run_model("Llama 3.1 70B", "meta-llama/llama-3.1-70b-instruct", 
                       os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
    if scores:
        results["Llama 3.1 70B"] = scores
    time.sleep(1)
    
    # Mistral via OpenRouter
    scores = run_model("Mistral Large 2", "mistralai/mistral-large-2407",
                       os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
    if scores:
        results["Mistral Large 2"] = scores
    
    # Save results
    output_file = PROJECT / "outputs" / "mfq30_fixes.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved fixes for {len(results)} models to {output_file}")

if __name__ == "__main__":
    main()
