#!/usr/bin/env python3
"""
Re-run Kimi K2.5 with better JSON parsing.
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
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_responses_filled_v4.csv"

def load_items():
    """Load all items."""
    items = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["item_id"], row["item_text"], row["format"]))
    return items

def generate_prompt(items):
    """Generate MFQ-30 prompt."""
    part1 = [(i, t) for i, t, f in items if f == "relevance"]
    part2 = [(i, t) for i, t, f in items if f == "judgment"]
    
    part1_text = "\n".join([f"{idx}. {text}" for idx, text in part1])
    part2_text = "\n".join([f"{idx}. {text}" for idx, text in part2])
    
    return f"""You will complete a moral foundations questionnaire in two parts.

=== PART 1: RELEVANCE ===
Rate each from 0 (not at all relevant) to 5 (extremely relevant):

{part1_text}

=== PART 2: AGREEMENT ===
Rate each from 0 (strongly disagree) to 5 (strongly agree):

{part2_text}

Respond with ONLY a JSON object where keys are item IDs (like CA01) and values are numbers 0-5. Example: {{\"CA01\": 4, "CA02\": 3}}"""

def call_kimi(prompt):
    """Call Kimi K2.5 via OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2.5",
        messages=[
            {"role": "system", "content": "You are completing a questionnaire. Output only valid JSON with no markdown formatting."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096,
        extra_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "MFQ-30 Survey"
        }
    )
    return response.choices[0].message.content

def parse_json(text):
    """Parse JSON with multiple attempts."""
    import re
    if not text:
        return None
    
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try fenced code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except:
            pass
    
    # Try finding JSON object
    m = re.search(r'(\{[^{}]*\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except:
            pass
    
    # Try line-by-line extraction
    result = {}
    for line in text.split('\n'):
        m = re.match(r'["\']?([A-Z]{2}\d{2})["\']?\s*[:=]\s*(\d+)', line.strip())
        if m:
            result[m.group(1)] = int(m.group(2))
    
    return result if result else None

def main():
    items = load_items()
    prompt = generate_prompt(items)
    
    print("[Kimi K2.5] Running...")
    
    try:
        raw = call_kimi(prompt)
        print(f"  Raw response length: {len(raw) if raw else 0}")
        
        scores = parse_json(raw)
        
        if scores:
            print(f"  ✓ Got {len(scores)} scores")
            
            # Append to output
            results = []
            for item_id, _, _ in items:
                score = scores.get(item_id, "")
                results.append({
                    "model": "Kimi K2.5",
                    "provider": "Moonshot",
                    "api_model": "kimi-k2.5",
                    "item_id": item_id,
                    "raw_score": str(score) if score != "" else "",
                })
            
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "provider", "api_model", "item_id", "raw_score"])
                writer.writerows(results)
            
            print(f"  ✓ Appended {len(results)} rows to {OUTPUT_CSV}")
        else:
            print(f"  ✗ Failed to parse JSON")
            print(f"  Raw: {raw[:500] if raw else 'None'}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    main()
