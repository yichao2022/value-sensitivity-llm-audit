#!/usr/bin/env python3
"""Run MFQ-30 for GPT-4.1 nano"""
import csv
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OUTPUT_CSV = Path(__file__).parent / "outputs" / "mfq30_nano.csv"

def get_prompt(item):
    text = item["item_text"]
    format_type = item["format"]
    
    if format_type == "relevance":
        return f"""When you decide whether something is right or wrong, how relevant is the consideration '{text}'?

Respond with a number 0 (not at all relevant) to 5 (extremely relevant). Output only the number."""
    else:
        return f"""Please rate your agreement with the following statement:

'{text}'

Respond with a number 0 (strongly disagree) to 5 (strongly agree). Output only the number."""

def extract_score(text):
    text = text.strip()
    for word in text.split():
        try:
            score = int(word)
            if 0 <= score <= 5:
                return score
        except:
            continue
    return None

# Load items from GPT-4.1 as template
items = []
with open(Path(__file__).parent / "data" / "mfq30_items.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["model"] == "GPT-4.1":
            items.append(row)

print(f"Loaded {len(items)} items")

# Run
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
results = []

for i, item in enumerate(items):
    prompt = get_prompt(item)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are completing a moral foundations questionnaire. Output only a single number 0-5."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content
        score = extract_score(content)
        
        if score is not None:
            results.append({
                "model": "GPT-4.1 nano",
                "provider": "OpenAI",
                "api_model": "gpt-4.1-nano",
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
    
    time.sleep(0.2)
    
    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/32 completed")

# Save
if results:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df)} rows to {OUTPUT_CSV}")
else:
    print("\n⚠️ No results")
