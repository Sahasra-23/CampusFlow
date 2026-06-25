import os
import json
import sys
import re
import requests
import time

# Reconfigure stdout/stderr to UTF-8 to handle emojis in Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Load .env file if it exists
def load_env():
    # Try parent directory first
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        # Try two levels up (workspace root)
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()

load_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for API keys
if not GEMINI_API_KEY and not GROQ_API_KEY:
    print("=" * 60)
    print("Warning: Neither GEMINI_API_KEY nor GROQ_API_KEY was found in environment or .env file.")
    print("To run actual LLM queries, please set one of these environment variables.")
    print("For now, the engine will run in validation-only dry-run mode using mock responses.")
    print("=" * 60)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
prompt_file = os.path.join(base_dir, 'system-prompt.txt')
cases_file = os.path.join(base_dir, 'test_cases.json')

# Load files
try:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
except FileNotFoundError:
    print(f"Error: system-prompt.txt not found at {prompt_file}")
    sys.exit(1)

try:
    with open(cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
except FileNotFoundError:
    print(f"Error: test_cases.json not found at {cases_file}")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: test_cases.json is not valid JSON")
    sys.exit(1)

def call_llm(message, idx):
    if GEMINI_API_KEY:
        # Call Gemini 2.5 Flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "parts": [{"text": message}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        # Free tier rate limit handling (5 RPM limit retry loop)
        for attempt in range(4):
            try:
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 429:
                    wait_time = 12.0  # sleep 12s on Gemini 429 to clear the rate window
                    print(f"  [Rate limit 429 encountered. Sleeping {wait_time}s and retrying (attempt {attempt+1}/4)...]")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                res_json = response.json()
                text = res_json['candidates'][0]['content']['parts'][0]['text']
                
                # Introduce a small baseline delay between successful requests to prevent rate limit spikes
                time.sleep(1.0)
                return text.strip()
            except Exception as e:
                if attempt == 3:  # last attempt failed
                    print(f"Gemini API Error: {e}")
                    if 'response' in locals() and response.text:
                        print(f"Response Details: {response.text}")
                    return None
                time.sleep(2.0)
                
    elif GROQ_API_KEY:
        # Call Groq Llama 3.3 70B
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(4):
            try:
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 429:
                    wait_time = 5.0
                    print(f"  [Rate limit 429 encountered. Sleeping {wait_time}s and retrying (attempt {attempt+1}/4)...]")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                res_json = response.json()
                text = res_json['choices'][0]['message']['content']
                time.sleep(1.0)
                return text.strip()
            except Exception as e:
                if attempt == 3:
                    print(f"Groq API Error: {e}")
                    if 'response' in locals() and response.text:
                        print(f"Response Details: {response.text}")
                    return None
                time.sleep(2.0)
    else:
        # Mock answers for Dry Run based on current date context (Thursday, June 25, 2026)
        mocks = {
            1: '{"isDeadline": true, "taskTitle": "Submit Computer Networks Lab File", "subject": "Computer Networks", "deadlineDate": "2026-06-26T16:00:00Z"}',
            2: '{"isDeadline": true, "taskTitle": "Submit Software Engineering Project Report", "subject": "Software Engineering", "deadlineDate": "2026-06-30T23:59:00Z"}',
            3: '{"isDeadline": true, "taskTitle": "Complete DBMS SQL Assignment", "subject": "DBMS", "deadlineDate": "2026-06-29T17:00:00Z"}',
            4: '{"isDeadline": true, "taskTitle": "CN Quiz", "subject": "Computer Networks", "deadlineDate": "2026-06-29T23:59:00Z"}',
            5: '{"isDeadline": true, "taskTitle": "OS Assignment", "subject": "Operating Systems", "deadlineDate": "2026-06-26T23:59:00Z"}',
            6: '{"isDeadline": true, "taskTitle": "Compiler Design Lab Internal", "subject": "Compiler Design", "deadlineDate": "2026-07-01T10:00:00Z"}',
            7: '{"isDeadline": false}',
            8: '{"isDeadline": false}',
            9: '{"isDeadline": false}',
            10: '{"isDeadline": false}'
        }
        return mocks.get(idx, '{"isDeadline": false}')

def validate_json_schema(data):
    if not isinstance(data, dict):
        return False, "Output is not a JSON object"
    
    if "isDeadline" not in data:
        return False, "Missing 'isDeadline' key"
    
    is_deadline = data["isDeadline"]
    if not isinstance(is_deadline, bool):
        return False, "'isDeadline' must be a boolean"
    
    if is_deadline:
        required_keys = {"taskTitle", "subject", "deadlineDate"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            return False, f"Missing required keys for deadline: {missing_keys}"
        
        if not isinstance(data["taskTitle"], str) or not data["taskTitle"].strip():
            return False, "'taskTitle' must be a non-empty string"
        if not isinstance(data["subject"], str) or not data["subject"].strip():
            return False, "'subject' must be a non-empty string"
        if not isinstance(data["deadlineDate"], str) or not data["deadlineDate"].strip():
            return False, "'deadlineDate' must be a non-empty string"
        
        # Check ISO 8601 format: YYYY-MM-DDTHH:mm:ssZ or YYYY-MM-DDTHH:mm:ss
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?$'
        if not re.match(iso_pattern, data["deadlineDate"]):
            return False, f"'deadlineDate' format invalid (got {data['deadlineDate']})"
            
        return True, "Valid deadline JSON"
    else:
        extra_keys = set(data.keys()) - {"isDeadline"}
        if extra_keys:
            return False, f"Extraneous keys found when isDeadline is false: {extra_keys}"
        return True, "Valid non-deadline JSON"

def run_tests():
    provider = "Gemini" if GEMINI_API_KEY else ("Groq" if GROQ_API_KEY else "MOCK (Dry Run)")
    print("=" * 60)
    print(f"CampusFlow QA Test Engine — Running using {provider} API")
    print("=" * 60)
    
    passed_count = 0
    total_count = len(test_cases)
    
    for idx, case in enumerate(test_cases, 1):
        input_text = case.get("input", "")
        expected_is_deadline = case.get("expected_isDeadline", False)
        
        print(f"\n[Test {idx}/{total_count}]")
        print(f"Input Chat: {input_text}")
        print(f"Expected isDeadline: {expected_is_deadline}")
        
        raw_output = call_llm(input_text, idx)
        
        if raw_output is None:
            print("Status: FAILED (LLM API error)")
            continue
            
        print(f"Raw LLM Output: {raw_output}")
        
        try:
            parsed = json.loads(raw_output)
            is_valid, msg = validate_json_schema(parsed)
            if is_valid:
                if parsed["isDeadline"] == expected_is_deadline:
                    print(f"Status: PASSED ({msg})")
                    passed_count += 1
                else:
                    print(f"Status: FAILED (Schema valid, but expected isDeadline={expected_is_deadline}, got {parsed['isDeadline']})")
            else:
                print(f"Status: FAILED (Schema error: {msg})")
        except json.JSONDecodeError:
            print("Status: FAILED (Invalid JSON syntax)")
            
    print("\n" + "=" * 60)
    print(f"Summary: {passed_count}/{total_count} tests passed.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
