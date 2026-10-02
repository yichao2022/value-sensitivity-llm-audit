#!/usr/bin/env python3
"""
Re-run failed models for MFQ-30.
"""

import csv
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

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

Respond with JSON only: {{\"CA01\": 4, ...}}"""

def call_anthropic(api_key, prompt):
    """Call Anthropic API - try different model names."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    # Try claude-opus-4-20250514
    try:
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2048,
            temperature=0,
            system="Output only valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"    opus-4-20250514 failed: {e}")
        # Try claude-3-opus-20240229
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2048,
                temperature=0,
                system="Output only valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e2:
            print(f"    3-opus-20240229 failed: {e2}")
            return None

def call_google(api_key, model_name, prompt):
    """Call Google Gemini API."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    
    # Map to correct model names
    if "pro" in model_name.lower():
        model_id = "gemini-2.5-pro"
    elif "flash" in model_name.lower():
        model_id = "gemini-2.5-flash"
    else:
        model_id = "gemini-2.5-flash"
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=4096,
                system_instruction="Output only valid JSON."
            )
        )
        return response.text
    except Exception as e:
        print(f"    {model_id} failed: {e}")
        return None

def call_openrouter(api_key, model_id, prompt):
    """Call OpenRouter API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    {model_id} failed: {e}")
        return None

def parse_json(text):
    """Parse JSON from response."""
    import re
    if not text:
        return None
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            text = m.group(0).strip()
    try:
        return json.loads(text)
    except:
        return None

def main():
    items = load_items()
    prompt = generate_prompt(items)
    
    # Models to re-run
    models_to_run = [
        ("Claude Opus 4", "Anthropic", None),
        ("Gemini 2.5 Pro", "Google", "gemini-2.5-pro"),
        ("Kimi K2.5", "OpenRouter", "moonshotai/kimi-k2.5"),
    ]
    
    results = []
    
    for model_name, provider, model_id in models_to_run:
        print(f"\n[{model_name}]...")
        
        raw = None
        
        if provider == "Anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                raw = call_anthropic(api_key, prompt)
        elif provider == "Google":
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                raw = call_google(api_key, model_id, prompt)
        elif provider == "OpenRouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                raw = call_openrouter(api_key, model_id, prompt)
        
        scores = parse_json(raw)
        
        if scores:
            print(f"  ✓ Got {len(scores)} scores")
            for item_id, _, _ in items:
                score = scores.get(item_id, "")
                results.append({
                    "model": model_name,
                    "provider": provider,
                    "api_model": model_id or "",
                    "item_id": item_id,
                    "raw_score": str(score) if score != "" else "",
                })
        else:
            print(f"  ✗ Failed")
            for item_id, _, _ in items:
                results.append({
                    "model": model_name,
                    "provider": provider,
                    "api_model": model_id or "",
                    "item_id": item_id,
                    "raw_score": "",
                })
        
        time.sleep(2)
    
    # Append to output
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "provider", "api_model", "item_id", "raw_score"])
        writer.writerows(results)
    
    print(f"\nDone. Appended {len(results)} rows.")

if __name__ == "__main__":
    main()
