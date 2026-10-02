#!/usr/bin/env python3
"""
Fix remaining models with nested JSON handling.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT = Path(__file__).resolve().parent

def call_api(api_key, model, prompt, base_url=None):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output only flat JSON with item IDs as keys."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096
    )
    return resp.choices[0].message.content

def parse_nested_json(text):
    """Handle nested JSON formats."""
    import re
    if not text:
        return None
    
    text = text.strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        # Flatten if nested
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result.update(value)
            elif isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                result[key] = int(value) if isinstance(value, int) else int(value)
        return result if len(result) >= 10 else None
    except:
        pass
    
    # Try extracting all item-score pairs
    result = {}
    for match in re.finditer(r'["\']?([A-Z]{2}\d{2})["\']?\s*[:=]\s*(\d+)', text):
        result[match.group(1)] = int(match.group(2))
    
    return result if len(result) >= 10 else None

def run_model(name, model_id, api_key, base_url=None):
    # Simple prompt
    prompt = """Rate these 32 MFQ-30 items on 0-5 scale.

Relevance items (0-5): CA01-CA06, FA07-FA12, LO13-LO18, AU19-AU24, SA25-SA30
Agreement items (0-5): CA31-CA32, FA33-FA35, LO36-LO38, AU39-AU41, SA42-SA44

Respond with flat JSON: {\"CA01\": 4, "CA02\": 3, ...}"""
    
    print(f"[{name}]...", end=" ", flush=True)
    try:
        raw = call_api(api_key, model_id, prompt, base_url)
        scores = parse_nested_json(raw)
        
        if scores and len(scores) >= 20:
            print(f"✓ {len(scores)} scores")
            return scores
        else:
            print(f"✗ Only {len(scores) if scores else 0} scores")
            print(f"  Raw: {raw[:300] if raw else 'None'}")
            return None
    except Exception as e:
        print(f"✗ {e}")
        return None

def main():
    results = {}
    
    # GPT-4.1 nano
    scores = run_model("GPT-4.1 nano", "gpt-4.1-nano", os.getenv("OPENAI_API_KEY"))
    if scores:
        results["GPT-4.1 nano"] = scores
    time.sleep(1)
    
    # Mistral
    scores = run_model("Mistral Large 2", "mistralai/mistral-large-2407",
                       os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")
    if scores:
        results["Mistral Large 2"] = scores
    
    # Load existing fixes and merge
    fix_file = PROJECT / "outputs" / "mfq30_fixes.json"
    if fix_file.exists():
        with open(fix_file) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    
    with open(fix_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal fixed models: {len(results)}")

if __name__ == "__main__":
    main()
