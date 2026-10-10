#!/usr/bin/env python3
"""Run H3 placebo experiment - simplified version."""
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()
HOME = Path(__file__).resolve().parent

# Load plan
plan_path = HOME / "data" / "h3_placebo_plan.csv"
output_path = HOME / "outputs" / "h3_placebo_results.csv"

print(f"Loading plan: {plan_path}")
with open(plan_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total runs: {len(rows)}")
print(f"First row model: {rows[0]['model']}, provider: {rows[0]['provider']}")

# Prepare output
results = []

# Run first 5 tests
for i, row in enumerate(rows[:5]):
    print(f"\nRun {i+1}/5: {row['model']}")
    
    provider = row['provider']
    api_model = row['api_model']
    
    # Get API config
    if provider == 'OpenAI':
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = None
    elif provider == 'Meta':
        api_key = os.getenv('OPENROUTER_API_KEY')
        base_url = "https://openrouter.ai/api/v1"
    elif provider == 'Mistral':
        api_key = os.getenv('OPENROUTER_API_KEY')
        base_url = "https://openrouter.ai/api/v1"
    elif provider == 'Qwen':
        api_key = os.getenv('DASHSCOPE_API_KEY')
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    else:
        print(f"Unknown provider: {provider}")
        continue
    
    if not api_key:
        print(f"No API key for {provider}")
        continue
    
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": row['system_prompt']},
                {"role": "user", "content": row['user_prompt']}
            ],
            temperature=0,
            max_tokens=500
        )
        
        result = {
            'run_id': row['run_id'],
            'model': row['model'],
            'frame': row['frame'],
            'raw_response': response.choices[0].message.content,
            'error': ''
        }
        results.append(result)
        print(f"Success: {response.choices[0].message.content[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        results.append({
            'run_id': row['run_id'],
            'model': row['model'],
            'frame': row['frame'],
            'raw_response': '',
            'error': str(e)
        })
    
    time.sleep(1)

# Save
print(f"\nSaving {len(results)} results to {output_path}")
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run_id', 'model', 'frame', 'raw_response', 'error'])
    writer.writeheader()
    writer.writerows(results)

print("Done")
