#!/usr/bin/env python3
"""
Send email to Dr. Chandra with API notification
"""
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
MAIN_TABLE_ID = None  # Will be set when calling the script

def send_notification_email(main_table_id):
    """Call SendMessageToConfirmPerson API after sending email"""
    import requests
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Warning: API_KEY not found in environment")
        return
    
    url = "https://api.yourservice.com/v1/SendMessageToConfirmPerson"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "mainTableId": main_table_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        print(f"API notification sent successfully for mainTableId: {main_table_id}")
    except Exception as e:
        print(f"Failed to send API notification: {e}")

def main():
    # Read the email content
    email_file = Path(__file__).parent / "email_to_chandra.txt"
    if not email_file.exists():
        print(f"Error: {email_file} not found")
        sys.exit(1)
    
    content = email_file.read_text()
    print("Email content ready:")
    print("=" * 50)
    print(content)
    print("=" * 50)
    print("\nTo send this email and notify the API:")
    print(f"1. Send the email above to Dr. Chandra")
    print(f"2. Run: python3 send_chandra_email.py --notify {MAIN_TABLE_ID or '<your-table-id>'}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--notify" and len(sys.argv) > 2:
        table_id = sys.argv[2]
        send_notification_email(table_id)
    else:
        main()
