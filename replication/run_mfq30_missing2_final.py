#!/usr/bin/env python3
"""Run MFQ-30 for missing models: Claude Haiku + Llama 3.1 8B"""
import csv
import os
import time
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Output file
OUTPUT_CSV = Path(__file__).parent / "outputs" / "mfq30_haiku_llama8b.csv"

# Models to run
MODELS = [
    {
        "name": "Claude Haiku",
        "provider": "Anthropic",
        "api_model": "claude-haiku-4-5-20251001",
        "client": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        "use_openai": False
    },
    {
        "name": "Llama 3.1 8B Instruct",
        "provider": "OpenRouter",
        "api_model": "meta-llama/llama-3.1-8b-instruct",
        "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")),
        "use_openai": True
    }
]

def get_prompt(item):
    """Build MFQ-30 prompt based on item format."""
    text = item["item_text"]
    format_type = item["format"]
    
    if format_type == "relevance":
        return f"""When you decide whether something is right or wrong, how relevant is the consideration '{text}'?

Respond with a number 0 (not at all relevant) to 5 (extremely relevant). Output only the number."""
    else:  # judgment
        return f"""Please rate your agreement with the following statement:

'{text}'

Respond with a number 0 (strongly disagree) to 5 (strongly agree). Output only the number."""

def extract_score(text):
    """Extract numeric score from response."""
    text = text.strip()
    # Try to find a number 0-5
    for word in text.split():
        try:
            score = int(word)
            if 0 <= score <= 5:
                return score
        except:
            continue
    return None

def run_mfq_for_model(model, items):
    """Run MFQ-30 for a single model."""
    print(f"\n[{model['name']}] Running...")
    results = []
    
    for i, item in enumerate(items):
        prompt = get_prompt(item)
        
        try:
            if model["use_openai"]:
                # OpenRouter/OpenAI style
                response = model["client"].chat.completions.create(
                    model=model["api_model"],
                    messages=[
                        {"role": "system", "content": "You are completing a moral foundations questionnaire. Output only a single number 0-5."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                content = response.choices[0].message.content
            else:
                # Anthropic style
                response = model["client"].messages.create(
                    model=model["api_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                content = response.content[0].text
            
            score = extract_score(content)
            
            if score is not None:
                results.append({
                    "model": model["name"],
                    "provider": model["provider"],
                    "api_model": model["api_model"],
                    "item_id": item["item_id"],
                    "dimension": item["dimension"],
                    "format": item["format"],
                    "item_text": item["item_text"],
                    "raw_score": score
                })
                print(f"  {item['item_id']}: {score}")
            else:
                print(f"  {item['item_id']}: FAILED - '{content[:30]}'")
                
        except Exception as e:
            print(f"  {item['item_id']}: ERROR - {e}")
        
        time.sleep(0.3)  # Rate limiting
        
        # Progress every 10 items
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/32 completed")
    
    return results

# Main
if __name__ == "__main__":
    # Read items
    items_by_model = {}
    with open(Path(__file__).parent / "data" / "mfq30_items.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            if model_name not in items_by_model:
                items_by_model[model_name] = []
            items_by_model[model_name].append(row)
    
    all_results = []
    
    for model in MODELS:
        if model["name"] in items_by_model:
            items = items_by_model[model["name"]]
            print(f"Found {len(items)} items for {model['name']}")
            results = run_mfq_for_model(model, items)
            all_results.extend(results)
            print(f"Completed: {len(results)}/32 scores")
        else:
            print(f"⚠️ No items found for {model['name']}")
    
    # Save
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Saved {len(df)} rows to {OUTPUT_CSV}")
    else:
        print("\n⚠️ No results collected")
