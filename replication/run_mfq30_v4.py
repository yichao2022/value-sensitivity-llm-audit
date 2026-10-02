#!/usr/bin/env python3
"""
Canonical MFQ-30 - Fixed version with correct model IDs and Google API.
"""

import csv
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_canonical_items.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_responses_filled_v4.csv"


def load_input():
    """Read input CSV."""
    rows = []
    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_model(rows):
    """Group rows by model."""
    groups = defaultdict(lambda: {"model": "", "provider": "", "api_model": "", "items": []})
    for r in rows:
        key = r["model"]
        groups[key]["model"] = r["model"]
        groups[key]["provider"] = r["provider"]
        groups[key]["api_model"] = r["api_model"]
        groups[key]["items"].append((r["item_id"], r["dimension"], r["format"], r["item_text"]))
    return list(groups.values())


def generate_mfq_prompt(items):
    """Generate two-part MFQ-30 prompt."""
    part1_items = [(i, t) for i, d, f, t in items if f == "relevance"]
    part2_items = [(i, t) for i, d, f, t in items if f == "judgment"]
    
    part1_text = "\n".join([f"{item_id}. {text}" for item_id, text in part1_items])
    part2_text = "\n".join([f"{item_id}. {text}" for item_id, text in part2_items])
    
    return f"""You will complete a moral foundations questionnaire in two parts.

=== PART 1: RELEVANCE ===
When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?

Rate each consideration from 0 (not at all relevant) to 5 (extremely relevant).

{part1_text}
=== PART 2: AGREEMENT ===
Please indicate your agreement with the following statements.

Rate each statement from 0 (strongly disagree) to 5 (strongly agree).

{part2_text}

=== INSTRUCTIONS ===
Provide your responses as a JSON object with item IDs as keys and numeric values (0-5) as values. Example: {{\"CA01\": 4, "CA02\": 3, ...}}

Respond only with the JSON object, no additional text."""


def call_openai(api_key, api_model, prompt, base_url=None):
    """Call OpenAI API."""
    from openai import OpenAI
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": "You are completing a moral foundations questionnaire. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2048
    )
    return response.choices[0].message.content


def call_anthropic(api_key, api_model, prompt):
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=api_model,
        max_tokens=2048,
        temperature=0,
        system="You are completing a moral foundations questionnaire. Output only valid JSON.",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def call_google(api_key, api_model, prompt):
    """Call Google Gemini API using new google.genai package."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=api_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
            system_instruction="You are completing a moral foundations questionnaire. Output only valid JSON."
        )
    )
    return response.text


def parse_json_response(text):
    """Extract JSON from response."""
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
    except json.JSONDecodeError:
        return None


def main():
    print("=" * 60)
    print("Canonical MFQ-30 (Fixed - v4)")
    print("=" * 60)
    
    rows = load_input()
    models = group_by_model(rows)
    
    print(f"\nTotal models: {len(models)}")
    print(f"Items per model: 32")
    print(f"Scale: 0-5 (canonical MFQ-30)")
    
    results = []
    
    for i, group in enumerate(models, 1):
        model_name = group["model"]
        provider = group["provider"]
        api_model = group["api_model"]
        items = group["items"]
        
        print(f"\n[{i}/{len(models)}] {model_name} ({provider})...")
        
        try:
            prompt = generate_mfq_prompt(items)
            raw = None
            
            # Route to appropriate API
            if provider == "Anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing ANTHROPIC_API_KEY")
                raw = call_anthropic(api_key, api_model, prompt)
                
            elif provider == "Google":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing GEMINI_API_KEY")
                # Use standard model names for google.genai
                if "thinking" in api_model:
                    model_to_use = "gemini-2.5-flash-thinking-exp"
                elif "flash" in api_model.lower():
                    model_to_use = "gemini-2.5-flash"
                else:
                    model_to_use = "gemini-2.5-pro"
                raw = call_google(api_key, model_to_use, prompt)
                
            elif provider == "OpenAI":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing OPENAI_API_KEY")
                raw = call_openai(api_key, api_model, prompt)
                
            elif provider == "DeepSeek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing DEEPSEEK_API_KEY")
                raw = call_openai(api_key, api_model, prompt, base_url="https://api.deepseek.com")
                
            elif provider == "Qwen":
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing DASHSCOPE_API_KEY")
                raw = call_openai(api_key, "qwen-max", prompt, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                
            else:
                # OpenRouter providers (Meta, Moonshot, Cohere, Mistral)
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("Missing OPENROUTER_API_KEY")
                # Map provider to correct OpenRouter model ID
                model_map = {
                    "Meta": "meta-llama/llama-3.1-70b-instruct",
                    "Moonshot": "moonshotai/kimi-k2.5",
                    "Cohere": "cohere/command-r-plus-08-2024",  # Fixed
                    "Mistral": "mistralai/mistral-large-2407",  # Fixed
                }
                model_to_use = model_map.get(provider, api_model)
                raw = call_openai(api_key, model_to_use, prompt, base_url="https://openrouter.ai/api/v1")
            
            scores = parse_json_response(raw)
            
            if scores:
                print(f"  ✓ Got {len(scores)} scores")
                for item_id, dim, fmt, text in items:
                    score = scores.get(item_id, "")
                    results.append({
                        "model": model_name,
                        "provider": provider,
                        "api_model": api_model,
                        "item_id": item_id,
                        "dimension": dim,
                        "format": fmt,
                        "item_text": text,
                        "raw_score": str(score) if score != "" else "",
                    })
            else:
                print(f"  ✗ Failed to parse JSON")
                for item_id, dim, fmt, text in items:
                    results.append({"model": model_name, "provider": provider, "api_model": api_model,
                                    "item_id": item_id, "dimension": dim, "format": fmt, "item_text": text, "raw_score": ""})
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            for item_id, dim, fmt, text in items:
                results.append({"model": model_name, "provider": provider, "api_model": api_model,
                                "item_id": item_id, "dimension": dim, "format": fmt, "item_text": text, "raw_score": ""})
        
        time.sleep(1)
    
    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "provider", "api_model", "item_id", "dimension", "format", "item_text", "raw_score"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} rows → {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
