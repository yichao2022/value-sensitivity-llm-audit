#!/usr/bin/env python3
"""
Kling AI (可灵) Video Generation — Text-to-Video & Image-to-Video

Usage:
  python3 kling_video.py --prompt "一只小拉布拉多在草地上奔跑" --duration 5
  python3 kling_video.py --prompt "..." --image /path/to/image.png --duration 5

Environment:
  KLING_ACCESS_KEY  or pass --access-key
  KLING_SECRET_KEY  or pass --secret-key
"""

import time, json, os, sys, argparse, requests
from pathlib import Path

API_BASE = "https://api-beijing.klingai.com"

def encode_jwt_token(ak: str, sk: str) -> str:
    import jwt
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,   # 30 min
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(payload, sk, headers=headers)

def create_text2video(ak: str, sk: str, prompt: str, duration: int = 5,
                      mode: str = "pro", aspect_ratio: str = "16:9",
                      model: str = "kling-v2-6", sound: str = "on") -> dict:
    """Create a text-to-video task. Returns task_id."""
    token = encode_jwt_token(ak, sk)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model_name": model,
        "prompt": prompt,
        "duration": str(duration),
        "mode": mode,
        "aspect_ratio": aspect_ratio,
        "sound": sound,
    }
    resp = requests.post(f"{API_BASE}/v1/videos/text2video",
                         headers=headers, json=payload)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"API error: {data.get('message', data)}")
    return data["data"]

def create_image2video(ak: str, sk: str, prompt: str, image_path: str,
                       duration: int = 5, mode: str = "pro",
                       model: str = "kling-v2-6") -> dict:
    """Create an image-to-video task. Returns task_id."""
    token = encode_jwt_token(ak, sk)
    headers = {"Authorization": f"Bearer {token}"}
    
    from PIL import Image
    import base64, io
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    payload = {
        "model_name": model,
        "prompt": prompt,
        "duration": str(duration),
        "mode": mode,
        "image_base64": b64,
    }
    resp = requests.post(f"{API_BASE}/v1/videos/image2video",
                         headers=headers, json=payload)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"API error: {data.get('message', data)}")
    return data["data"]

def poll_task(ak: str, sk: str, task_id: str, poll_interval: int = 3,
              timeout: int = 300) -> dict:
    """Poll until task completes or fails."""
    token = encode_jwt_token(ak, sk)
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{API_BASE}/v1/videos/text2video/{task_id}",
                            headers=headers)
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Query error: {data.get('message', data)}")
        status = data["data"]["task_status"]
        if status == "succeed":
            return data["data"]
        elif status == "failed":
            msg = data["data"].get("task_status_msg", "Unknown failure")
            raise RuntimeError(f"Task failed: {msg}")
        print(f"  Status: {status}, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    raise TimeoutError(f"Task did not complete within {timeout}s")

def download_video(url: str, output_path: str):
    """Download video from URL."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Kling AI Video Generator")
    parser.add_argument("--prompt", required=True, help="Text prompt for video")
    parser.add_argument("--image", help="Path to image for image-to-video (optional)")
    parser.add_argument("--duration", type=int, default=5, help="Video duration in seconds (3-15)")
    parser.add_argument("--mode", choices=["std", "pro", "4k"], default="pro", help="Quality mode")
    parser.add_argument("--ratio", choices=["16:9", "9:16", "1:1"], default="16:9", help="Aspect ratio")
    parser.add_argument("--model", default="kling-v2-6", help="Model name")
    parser.add_argument("--sound", choices=["on", "off"], default="on", help="Background sound")
    parser.add_argument("--output", default=None, help="Output path (default: ~/Desktop/kling_video_<timestamp>.mp4)")
    parser.add_argument("--access-key", help="Kling Access Key")
    parser.add_argument("--secret-key", help="Kling Secret Key")
    parser.add_argument("--no-download", action="store_true", help="Skip download, just print URL")
    
    args = parser.parse_args()
    
    ak = args.access_key or os.environ.get("KLING_ACCESS_KEY")
    sk = args.secret_key or os.environ.get("KLING_SECRET_KEY")
    if not ak or not sk:
        print("ERROR: Set KLING_ACCESS_KEY and KLING_SECRET_KEY or pass --access-key / --secret-key")
        sys.exit(1)
    
    # Create task
    if args.image:
        print(f"Creating image-to-video task...")
        print(f"  Image: {args.image}")
        print(f"  Prompt: {args.prompt}")
        result = create_image2video(ak, sk, args.prompt, args.image,
                                    duration=args.duration, mode=args.mode, model=args.model)
    else:
        print(f"Creating text-to-video task...")
        print(f"  Prompt: {args.prompt}")
        print(f"  Duration: {args.duration}s, Mode: {args.mode}, Ratio: {args.ratio}")
        print(f"  Model: {args.model}, Sound: {args.sound}")
        result = create_text2video(ak, sk, args.prompt, duration=args.duration,
                                   mode=args.mode, aspect_ratio=args.ratio,
                                   model=args.model, sound=args.sound)
    
    task_id = result["task_id"]
    print(f"  Task ID: {task_id}")
    print(f"  Initial status: {result['task_status']}")
    
    # Poll
    print("Polling for completion...")
    result = poll_task(ak, sk, task_id)
    
    video_info = result["task_result"]["videos"][0]
    video_url = video_info["url"]
    video_duration = video_info.get("duration", "?")
    print(f"\n✅ Video generated! Duration: {video_duration}s")
    print(f"  URL: {video_url}")
    
    if not args.no_download:
        output = args.output or str(Path.home() / "Desktop" / f"kling_video_{int(time.time())}.mp4")
        download_video(video_url, output)
    else:
        print("  (--no-download, skipping download)")

if __name__ == "__main__":
    main()
