#!/usr/bin/env python3
"""
Complete MFQ-30 runner - 15 models.
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
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_15models.csv"

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

PART 1 - RELEVANCE (0=not at all, 5=extremely):
{part1_text}

PART 2 - AGREEMENT (0=strongly disagree, 5=strongly agree):
{part2_text}

Respond with JSON: {{\"CA01\": 4, ...}}"""

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

def call_google(api_key, model, prompt):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=4096)
    )
    return resp.text

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
    
    # 15 models
    models = [
        ("GPT-4.1", "OpenAI", "gpt-4.1", lambda p: call_openai(os.getenv("OPENAI_API_KEY"), "gpt-4.1", p)),
        ("GPT-4.1 mini", "OpenAI", "gpt-4.1-mini", lambda p: call_openai(os.getenv("OPENAI_API_KEY"), "gpt-4.1-mini", p)),
        ("GPT-4.1 nano", "OpenAI", "gpt-4.1-nano", lambda p: call_openai(os.getenv("OPENAI_API_KEY"), "gpt-4.1-nano", p)),
        ("GPT-4o", "OpenAI", "gpt-4o", lambda p: call_openai(os.getenv("OPENAI_API_KEY"), "gpt-4o", p)),
        ("Claude Sonnet 4.6", "Anthropic", "claude-sonnet-4-6", lambda p: call_anthropic(os.getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-6", p)),
        ("Claude 3.5 Sonnet", "Anthropic", "claude-3-5-sonnet-20241022", lambda p: call_anthropic(os.getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-20241022", p)),
        ("Gemini 2.5 Flash", "Google", "gemini-2.5-flash", lambda p: call_google(os.getenv("GEMINI_API_KEY"), "gemini-2.5-flash", p)),
        ("Gemini 2.5 Pro", "Google", "gemini-2.5-pro", lambda p: call_google(os.getenv("GEMINI_API_KEY"), "gemini-2.5-pro", p)),
        ("Gemini 1.5 Flash", "Google", "gemini-1.5-flash", lambda p: call_google(os.getenv("GEMINI_API_KEY"), "gemini-1.5-flash", p)),
        ("DeepSeek-V3.2", "DeepSeek", "deepseek-flash", lambda p: call_openai(os.getenv("DEEPSEEK_API_KEY"), "deepseek-flash", p, "https://api.deepseek.com")),
        ("Qwen3.6-72B", "Qwen", "qwen-max", lambda p: call_openai(os.getenv("DASHSCOPE_API_KEY"), "qwen-max", p, "https://dashscope.aliyuncs.com/compatible-mode/v1")),
        ("Llama 3.1 70B", "Meta", "meta-llama/llama-3.1-70b-instruct", lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "meta-llama/llama-3.1-70b-instruct", p, "https://openrouter.ai/api/v1")),
        ("Command R+", "Cohere", "cohere/command-r-plus-08-2024", lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "cohere/command-r-plus-08-2024", p, "https://openrouter.ai/api/v1")),
        ("Mistral Large 2", "Mistral", "mistralai/mistral-large-2407", lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "mistralai/mistral-large-2407", p, "https://openrouter.ai/api/v1")),
        ("Kimi K2.5", "Moonshot", "moonshotai/kimi-k2.5", lambda p: call_openai(os.getenv("OPENROUTER_API_KEY"), "moonshotai/kimi-k2.5", p, "https://openrouter.ai/api/v1")),
    ]
    
    results = []
    
    for name, provider, api_model, caller in models:
        print(f"[{name}]...", end=" ", flush=True)
        try:
            raw = caller(prompt)
            scores = parse_json(raw)
            if scores:
                print(f"✓ {len(scores)} scores")
                for item in items:
                    score = scores.get(item['item_id'], '')
                    results.append({
                        'model': name, 'provider': provider, 'api_model': api_model,
                        'item_id': item['item_id'], 'dimension': item['dimension'],
                        'format': item['format'], 'item_text': item['item_text'],
                        'raw_score': str(score) if score != '' else ''
                    })
            else:
                print("✗ JSON failed")
                for item in items:
                    results.append({
                        'model': name, 'provider': provider, 'api_model': api_model,
                        'item_id': item['item_id'], 'dimension': item['dimension'],
                        'format': item['format'], 'item_text': item['item_text'],
                        'raw_score': ''
                    })
        except Exception as e:
            print(f"✗ {e}")
            for item in items:
                results.append({
                    'model': name, 'provider': provider, 'api_model': api_model,
                    'item_id': item['item_id'], 'dimension': item['dimension'],
                    'format': item['format'], 'item_text': item['item_text'],
                    'raw_score': ''
                })
        time.sleep(1)
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'provider', 'api_model', 'item_id', 'dimension', 'format', 'item_text', 'raw_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDone: {len(results)} rows")

if __name__ == "__main__":
    main()
