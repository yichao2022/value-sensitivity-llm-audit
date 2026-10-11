#!/usr/bin/env python3
"""Run MFQ-30 for missing models: Claude Haiku + Llama 3.1 8B"""
import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# API keys
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Models to run
MODELS = [
    {
        "name": "Claude Haiku 4.5",
        "provider": "anthropic",
        "api_model": "claude-haiku-4-5-20251001",
        "client": OpenAI(base_url="https://api.anthropic.com/v1", api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
    },
    {
        "name": "Llama 3.1 8B",
        "provider": "openrouter", 
        "api_model": "meta-llama/llama-3.1-8b-instruct",
        "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY) if OPENROUTER_KEY else None
    }
]

# MFQ-30 items
ITEMS = [
    {"id": "CA01", "dimension": "CA", "format": "relevance", "text": "Whether or not someone suffered emotionally"},
    {"id": "CA02", "dimension": "CA", "format": "relevance", "text": "Whether or not someone cared for someone weak or vulnerable"},
    {"id": "CA03", "dimension": "CA", "format": "relevance", "text": "Whether or not someone was cruel"},
    {"id": "CA04", "dimension": "CA", "format": "judgment", "text": "Compassion for those who are suffering is the most crucial virtue."},
    {"id": "CA05", "dimension": "CA", "format": "judgment", "text": "When the government makes laws, the number one principle should be ensuring that everyone is treated fairly."},
    {"id": "CA06", "dimension": "CA", "format": "judgment", "text": "It is more important to be empathetic than to be just."},
]

# Add remaining 24 items...
ALL_ITEMS = ITEMS  # Placeholder - should include all 30 items

def run_mfq_for_model(model):
    """Run MFQ-30 for a single model."""
    if not model["client"]:
        print(f"⚠️  No API key for {model['name']}, skipping")
        return []
    
    results = []
    for item in ALL_ITEMS:
        # Build prompt
        if item["format"] == "relevance":
            prompt = f"When you decide whether something is right or wrong, how relevant is the consideration '{item['text']}'?\n\nRespond with a number 0 (not at all relevant) to 5 (extremely relevant)."
        else:
            prompt = f"Please rate your agreement with the following statement:\n\n'{item['text']}'\n\nRespond with a number 0 (strongly disagree) to 5 (strongly agree)."
        
        try:
            response = model["client"].chat.completions.create(
                model=model["api_model"],
                messages=[
                    {"role": "system", "content": "You are completing a moral foundations questionnaire. Output only a single number 0-5."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=10
            )
            
            content = response.choices[0].message.content.strip()
            # Extract number
            score = None
            for word in content.split():
                try:
                    score = int(word)
                    if 0 <= score <= 5:
                        break
                except:
                    continue
            
            if score is not None:
                results.append({
                    "model": model["name"],
                    "provider": model["provider"],
                    "api_model": model["api_model"],
                    "item_id": item["id"],
                    "dimension": item["dimension"],
                    "format": item["format"],
                    "item_text": item["text"],
                    "raw_score": score
                })
                print(f"  {item['id']}: {score}")
            else:
                print(f"  {item['id']}: failed to parse")
                
        except Exception as e:
            print(f"  {item['id']}: error - {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return results

# Run for both models
all_results = []
for model in MODELS:
    print(f"\n[{model['name']}] ...")
    results = run_mfq_for_model(model)
    all_results.extend(results)
    print(f"  Got {len(results)}/32 scores")

# Save
if all_results:
    df = pd.DataFrame(all_results)
    output_path = Path(__file__).parent / "outputs" / "mfq30_missing_2models.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_path}")
else:
    print("\n⚠️ No results collected")
