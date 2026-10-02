#!/usr/bin/env python3
"""
Add Claude 3 Opus and Gemini 1.5 Pro.
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT = Path(__file__).resolve().parent

PROMPT = """Rate these 32 MFQ items (0-5 scale):

Relevance: CA01,CA02,CA03,CA04,CA05,CA06,FA07,FA08,FA09,FA10,FA11,FA12,LO13,LO14,LO15,LO16,LO17,LO18,AU19,AU20,AU21,AU22,AU23,AU24,SA25,SA26,SA27,SA28,SA29,SA30
Agreement: CA31,CA32,FA33,FA34,FA35,LO36,LO37,LO38,AU39,AU40,AU41,SA42,SA43,SA44

Reply with ONLY flat JSON: {\"CA01\":4,\"CA02\":5,...}"""

def call_anthropic(api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=4096, temperature=0,
        messages=[{"role": "user", "content": PROMPT}]
    )
    return resp.content[0].text

def call_google(api_key, model):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model, contents=PROMPT,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=4096)
    )
    return resp.text

def parse_scores(text):
    scores = {}
    for match in re.finditer(r'["\']?([A-Z]{2}\d{2})["\']?\s*:\s*(\d+)', text):
        scores[match.group(1)] = int(match.group(2))
    return scores

def main():
    results = {}
    
    print("[Claude 3 Opus]...", end=" ", flush=True)
    try:
        raw = call_anthropic(os.getenv("ANTHROPIC_API_KEY"), "claude-3-opus-20240229")
        scores = parse_scores(raw)
        if len(scores) >= 30:
            print(f"OK {len(scores)} scores")
            results["Claude 3 Opus"] = scores
        else:
            print(f"FAIL {len(scores)} scores")
    except Exception as e:
        print(f"ERROR {e}")
    
    print("[Gemini 1.5 Pro]...", end=" ", flush=True)
    try:
        raw = call_google(os.getenv("GEMINI_API_KEY"), "gemini-1.5-pro")
        scores = parse_scores(raw)
        if len(scores) >= 30:
            print(f"OK {len(scores)} scores")
            results["Gemini 1.5 Pro"] = scores
        else:
            print(f"FAIL {len(scores)} scores")
    except Exception as e:
        print(f"ERROR {e}")
    
    output = PROJECT / "outputs" / "mfq15_opus_pro.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} models")

if __name__ == "__main__":
    main()
