#!/usr/bin/env python3
"""
Fix GPT-4.1 nano and Mistral.
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT = Path(__file__).resolve().parent

PROMPT = """Rate these 32 MFQ items (0-5 scale):

Relevance: CA01,CA02,CA03,CA04,CA05,CA06,FA07,FA08,FA09,FA10,FA11,FA12,LO13,LO14,LO15,LO16,LO17,LO18,AU19,AU20,AU21,AU22,AU23,AU24,SA25,SA26,SA27,SA28,SA29,SA30
Agreement: CA31,CA32,FA33,FA34,FA35,LO36,LO37,LO38,AU39,AU40,AU41,SA42,SA43,SA44

Reply with ONLY flat JSON like: {\"CA01\":4,\"CA02\":5,...}"""

def call_openai(api_key, model):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        max_tokens=2048
    )
    return resp.choices[0].message.content

def call_openrouter(api_key, model):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        max_tokens=2048
    )
    return resp.choices[0].message.content

def parse_scores(text):
    scores = {}
    for match in re.finditer(r'["\']?([A-Z]{2}\d{2})["\']?\s*:\s*(\d+)', text):
        scores[match.group(1)] = int(match.group(2))
    return scores

def main():
    results = {}
    
    print("[GPT-4.1 nano]...", end=" ", flush=True)
    try:
        raw = call_openai(os.getenv("OPENAI_API_KEY"), "gpt-4.1-nano")
        scores = parse_scores(raw)
        if len(scores) >= 30:
            print(f"✓ {len(scores)} scores")
            results["GPT-4.1 nano"] = scores
        else:
            print(f"✗ {len(scores)} scores")
            print(f"  Raw: {raw[:200] if raw else 'None'}")
    except Exception as e:
        print(f"✗ {e}")
    
    print("[Mistral Large 2]...", end=" ", flush=True)
    try:
        raw = call_openrouter(os.getenv("OPENROUTER_API_KEY"), "mistralai/mistral-large-2407")
        scores = parse_scores(raw)
        if len(scores) >= 30:
            print(f"✓ {len(scores)} scores")
            results["Mistral Large 2"] = scores
        else:
            print(f"✗ {len(scores)} scores")
            print(f"  Raw: {raw[:200] if raw else 'None'}")
    except Exception as e:
        print(f"✗ {e}")
    
    output = PROJECT / "outputs" / "mfq30_additional_fixes.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFixed: {list(results.keys())}")

if __name__ == "__main__":
    main()
