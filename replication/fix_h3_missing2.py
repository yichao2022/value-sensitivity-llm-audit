#!/usr/bin/env python3
"""补跑 H3 placebo 缺失的 2 条 Mistral - 最终版"""
import csv
import os
import re
import time
from pathlib import Path

os.chdir('/Users/cary/Documents/value-sensitivity-llm-audit/replication')

RESULTS_CSV = Path('outputs/h3_placebo_results.csv')
PLAN_CSV = Path('data/h3_placebo_plan.csv')
MISSING_IDS = ['H3P_00584', 'H3P_00711']

def extract_willingness(text):
    if not text or not isinstance(text, str):
        return None
    # 优先匹配明确的 willingness 声明
    pattern = r'willingness\s+(?:to\s+proceed|to\s+be\s+vaccinated|score)\s*[:\-]?\s*(?:is\s+)?(?:approximately\s+)?(\d+)(?:\s*%)?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        n = int(match.group(1))
        if 0 <= n <= 100:
            return n
    # 回退：找 1-100 的数字
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    for num_str in numbers:
        n = int(num_str)
        if 1 <= n <= 100:
            return n
    # 最后包含 0
    for num_str in numbers:
        n = int(num_str)
        if n == 0:
            return n
    return None

def call_mistral(prompt):
    import requests
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return None, None
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://localhost',
        'X-Title': 'H3 Fix'
    }
    
    payload = {
        'model': 'mistralai/mistral-large',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
        'max_tokens': 500
    }
    
    for attempt in range(3):
        try:
            resp = requests.post('https://openrouter.ai/api/v1/chat/completions',
                               headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            data = resp.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                return extract_willingness(content), content
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}')
            time.sleep(2)
    return None, None

def main():
    print('=' * 50)
    print('补跑 H3 placebo 缺失的 2 条')
    print('=' * 50)
    
    # 加载 prompts
    prompts = {}
    with open(PLAN_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['run_id'] in MISSING_IDS:
                prompts[row['run_id']] = row['user_prompt']
    print(f'Loaded {len(prompts)} prompts')
    
    # 读取结果
    with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
        results = list(csv.DictReader(f))
    print(f'Current results: {len(results)} rows')
    
    # 补跑
    fixed = 0
    for run_id in MISSING_IDS:
        prompt = prompts.get(run_id, '')
        if not prompt:
            print(f'[{run_id}] No prompt found')
            continue
        
        print(f'\n[{run_id}] Calling Mistral...')
        willingness, response = call_mistral(prompt)
        
        if willingness is not None:
            print(f'  ✓ Got willingness = {willingness}')
            # 找到并更新行
            for row in results:
                if row.get('run_id') == run_id:
                    row['raw_response'] = response + f'\n\n[WILLINGNESS: {willingness}]'
                    fixed += 1
                    break
        else:
            print(f'  ✗ Failed')
        time.sleep(1.5)
    
    # 保存
    if fixed > 0:
        print(f'\nSaving {fixed} updates...')
        with open(RESULTS_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'✓ Done! Updated {RESULTS_CSV}')
    else:
        print('\n✗ No updates')

if __name__ == '__main__':
    main()
