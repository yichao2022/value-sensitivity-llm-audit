#!/usr/bin/env python3
"""补跑 H3 placebo 缺失的 2 条 Mistral 数据 - 从 plan 读取 prompt"""
import csv, os, re, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT / "outputs" / "h3_placebo_results.csv"
PLAN_CSV = PROJECT / "data" / "h3_placebo_plan.csv"

# 缺失的 2 条
MISSING_IDS = ["H3P_00584", "H3P_00711"]

def extract_willingness(text):
    """从响应中提取 willingness 数值 (0-100)"""
    if not text or not isinstance(text, str):
        return None
    # 模式: "willingness to proceed": 65 或单独数字
    patterns = [
        r'"?willingness\s+(?:to\s+proceed|to\s+be\s+vaccinated|score)"?[\s:]+(\d+)',
        r'[\[\(]?(\d{1,3})[\]\)]?\s*%?',
        r'\b(\d{1,3})\b'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            n = int(match)
            if 0 <= n <= 100:
                return n
    return None

def call_openrouter(user_prompt, model_id="mistralai/mistral-large", max_retries=3):
    """调用 OpenRouter API"""
    import requests
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "H3 Placebo Rerun"
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 429:
                wait_time = 2 ** attempt + 1
                print(f"    429 rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                willingness = extract_willingness(content)
                return willingness, content
            else:
                print(f"    No choices: {result.keys()}")
                
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None, None

def load_plan_prompts():
    """从 plan 文件加载所有 prompts"""
    prompts = {}
    with open(PLAN_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get('run_id')
            if run_id in MISSING_IDS:
                prompts[run_id] = row.get('user_prompt', '')
    return prompts

def main():
    print("=" * 60)
    print("补跑 H3 placebo 缺失的 2 条 Mistral")
    print("=" * 60)
    
    # 加载 plan 中的 prompts
    plan_prompts = load_plan_prompts()
    print(f"\n从 plan 加载了 {len(plan_prompts)} 条缺失 prompts")
    
    # 读取现有结果
    results = []
    with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    print(f"现有结果: {len(results)} 条")
    
    # 补跑每条缺失的
    updated_count = 0
    for run_id in MISSING_IDS:
        user_prompt = plan_prompts.get(run_id, '')
        if not user_prompt:
            print(f"\n[{run_id}] ✗ No prompt in plan")
            continue
        
        print(f"\n[{run_id}] Calling Mistral...")
        willingness, response_text = call_openrouter(user_prompt)
        
        if willingness is not None:
            print(f"  ✓ Success: willingness = {willingness}")
            # 更新数据 - 把 willingness 嵌入 raw_response
            for row in results:
                if row.get('run_id') == run_id:
                    # 在 raw_response 末尾添加 willingness 标记
                    marker = f"\n\n[EXTRACTED_WILLINGNESS: {willingness}]"
                    if marker not in row.get('raw_response', ''):
                        row['raw_response'] = row.get('raw_response', '') + marker
                    updated_count += 1
                    break
        else:
            print(f"  ✗ Failed to extract willingness")
            if response_text:
                print(f"    Preview: {response_text[:150]}...")
        
        time.sleep(1.5)  # 避免 rate limit
    
    # 保存结果
    if updated_count > 0:
        print(f"\n\n保存 {updated_count} 条更新...")
        with open(RESULTS_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ 完成! 已更新 {RESULTS_CSV}")
    else:
        print("\n✗ 没有成功更新")

if __name__ == "__main__":
    main()
