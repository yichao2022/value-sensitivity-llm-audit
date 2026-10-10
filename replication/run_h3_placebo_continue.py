#!/usr/bin/env python3
"""Continue H3 placebo experiment in background."""
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()
HOME = Path(__file__).resolve().parent

# Load plan
with open(HOME / "data" / "h3_placebo_plan.csv") as f:
    rows = list(csv.DictReader(f))

print(f"Total runs: {len(rows)}")

# Check for existing results
output_path = HOME / "outputs" / "h3_placebo_results.csv"
completed_ids = set()
if output_path.exists():
    with open(output_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed_ids.add(row['run_id'])
    print(f"Resuming: {len(completed_ids)} already completed")

# Run remaining
results = []
for i, row in enumerate(rows):
    if row['run_id'] in completed_ids:
        continue
    
    provider = row['provider']
    api_model = row['api_model']
    
    if provider == 'OpenAI':
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = None
    elif provider in ['Meta', 'Mistral']:
        api_key = os.getenv('OPENROUTER_API_KEY')
        base_url = "https://openrouter.ai/api/v1"
    elif provider == 'Qwen':
        api_key = os.getenv('DASHSCOPE_API_KEY')
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    else:
        continue
    
    if not api_key:
        continue
    
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        extra_headers = {}
        if base_url and "openrouter" in base_url:
            extra_headers = {
                "HTTP-Referer": "https://github.com/yichao2022/value-sensitivity-llm-audit",
                "X-Title": "LLM Policy Simulation Study"
            }
        
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": row['system_prompt']},
                {"role": "user", "content": row['user_prompt']}
            ],
            temperature=0,
            max_tokens=500,
            extra_headers=extra_headers
        )
        
        results.append({
            'run_id': row['run_id'],
            'model': row['model'],
            'frame': row['frame'],
            'raw_response': response.choices[0].message.content,
            'error': ''
        })
        
    except Exception as e:
        results.append({
            'run_id': row['run_id'],
            'model': row['model'],
            'frame': row['frame'],
            'raw_response': '',
            'error': str(e)
        })
    
    # Progress every 50
    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{len(rows)} (completed: {len(completed_ids) + len(results)})")
    
    # Save every 50
    if len(results) >= 50:
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_id', 'model', 'frame', 'raw_response', 'error'])
            if output_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} results")
        results = []
    
    time.sleep(0.3)

# Save remaining
if results:
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'model', 'frame', 'raw_response', 'error'])
        if output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(results)
    print(f"Saved final {len(results)} results")

# Count final
with open(output_path) as f:
    count = sum(1 for _ in f) - 1
print(f"Total completed: {count}")
print("Complete")
