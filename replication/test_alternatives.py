#!/usr/bin/env python3
"""
Try alternative models on OpenRouter.
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

Reply with ONLY flat JSON: {\"CA01\":4,\"CA02\":5,...}"""

def call_openrouter(api_key, model):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        max_tokens=4096
    )
    return resp.choices[0].message.content

def parse_scores(text):
    scores = {}
    for match in re.finditer(r'["\']?([A-Z]{2}\d{2})["\']?\s*:\s*(\d+)', text):
        scores[match.group(1)] = int(match.group(2))
    return scores

def test_model(name, model_id):
    print(f"[{name}]...", end=" ", flush=True)
    try:
        raw = call_openrouter(os.getenv("OPENROUTER_API_KEY"), model_id)
        scores = parse_scores(raw)
        if len(scores) >= 30:
            print(f"OK {len(scores)}")
            return scores
        else:
            print(f"FAIL {len(scores)}")
            return None
    except Exception as e:
        print(f"ERROR {str(e)[:50]}")
        return None

def main():
    results = {}
    
    # Try alternative models
    alternatives = [
        ("Gemma 2 27B", "google/gemma-2-27b-it"),
        ("Nous Hermes 2 Mixtral", "nousresearch/nous-hermes-2-mixtral-8x7b"),
        ("Qwen 2.5 72B", "qwen/qwen-2.5-72b-instruct"),
        ("Llama 3.3 70B", "meta-llama/llama-3.3-70b-instruct"),
    ]
    
    for name, model_id in alternatives:
        scores = test_model(name, model_id)
        if scores:
            results[name] = scores
    
    output = PROJECT / "outputs" / "mfq15_alternatives.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSuccess: {list(results.keys())}")

if __name__ == "__main__":
    main()
