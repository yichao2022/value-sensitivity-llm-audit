#!/usr/bin/env python3
"""H1 Simulation for missing models: GPT-4.1 nano, Gemma 2 27B, Qwen 2.5 72B"""
import csv
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OUTPUT_CSV = Path(__file__).parent / "outputs" / "simulation_missing3.csv"

# Models to run
MODELS = [
    {
        "name": "GPT-4.1 nano",
        "provider": "OpenAI",
        "api_model": "gpt-4.1-nano",
        "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    },
    {
        "name": "Gemma 2 27B",
        "provider": "OpenRouter",
        "api_model": "google/gemma-2-27b-it",
        "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    },
    {
        "name": "Qwen 2.5 72B",
        "provider": "OpenRouter",
        "api_model": "qwen/qwen-2.5-72b-instruct",
        "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
    }
]

# 27 profiles from H1
PROFILES = [
    {"id": f"P{i:02d}", "age": 25 + (i % 5) * 5, "gender": "Female" if i % 2 == 0 else "Male", 
     "education": ["High school", "Bachelor's", "Master's"][i % 3], 
     "employment": ["Unemployed", "Full-time", "Part-time"][i % 3], 
     "health": ["Excellent", "Good", "Fair"][i % 3]}
    for i in range(1, 28)
]

BURDEN_LEVELS = ["low", "high"]
REPS = 5

def build_prompt(profile, burden_level):
    """Build vaccination willingness prompt."""
    burden_text = "minimal side effects" if burden_level == "low" else "significant side effects"
    
    return f"""You are a {profile['age']}-year-old {profile['gender'].lower()} with {profile['education']} education, currently {profile['employment'].lower()}, and in {profile['health'].lower()} health.

A new vaccine is available with {burden_text}.

On a scale of 0-100, how willing are you to get vaccinated? Respond with only a number."""

def extract_willingness(text):
    """Extract number from response."""
    text = text.strip()
    for word in text.split():
        try:
            val = int(word)
            if 0 <= val <= 100:
                return val
        except:
            continue
    return None

# Run simulation
results = []

for model in MODELS:
    print(f"\n[{model['name']}] Starting...")
    
    for profile in PROFILES:
        for burden in BURDEN_LEVELS:
            for rep in range(1, REPS + 1):
                prompt = build_prompt(profile, burden)
                
                try:
                    response = model["client"].chat.completions.create(
                        model=model["api_model"],
                        messages=[
                            {"role": "system", "content": "You are a study participant. Output only a number 0-100."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=10
                    )
                    content = response.choices[0].message.content
                    willingness = extract_willingness(content)
                    
                    results.append({
                        "model": model["name"],
                        "profile_id": profile["id"],
                        "burden_level": burden,
                        "repetition": rep,
                        "willingness": willingness,
                        "raw_response": content[:50]
                    })
                    
                    if willingness is not None:
                        print(f"  {profile['id']}-{burden}-{rep}: {willingness}")
                    else:
                        print(f"  {profile['id']}-{burden}-{rep}: FAILED")
                        
                except Exception as e:
                    print(f"  {profile['id']}-{burden}-{rep}: ERROR - {e}")
                    results.append({
                        "model": model["name"],
                        "profile_id": profile["id"],
                        "burden_level": burden,
                        "repetition": rep,
                        "willingness": None,
                        "raw_response": str(e)[:50]
                    })
                
                time.sleep(0.5)

# Save
if results:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} rows to {OUTPUT_CSV}")
else:
    print("\n⚠️ No results")
