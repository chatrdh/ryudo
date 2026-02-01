import time
import requests
import subprocess
import os
import signal
import sys

def verify():
    print("🚀 Starting server in background...")
    # maximize output buffer to prevent hanging
    server_process = subprocess.Popen(
        [".venv/bin/python", "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    try:
        # Wait for server startup
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Trigger workflow
        print("🔌 Triggering workflow via API...")
        try:
            resp = requests.post("http://localhost:8000/api/start")
            print(f"API Response: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"❌ Failed to call API: {e}")
            return False
            
        # Wait for agents to run (they have sleep delays)
        print("🤖 Waiting for agents to reason (15s)...")
        time.sleep(15)
        
        # Check logs
        log_file = "output/llm_reasoning.md"
        if not os.path.exists(log_file):
            print("❌ Log file not found!")
            return False
            
        with open(log_file, "r") as f:
            content = f.read()
            
        # Check for recent timestamps
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in content:
            print(f"✅ Found today's logs ({today})!")
            
            # Check for Gemini model usage
            if "gemini-3-flash-preview" in content:
                 print("✅ Found Gemini model usage!")
                 return True
            else:
                 print("❌ Gemini model NOT found in logs!")
                 # Print last 500 chars
                 print(content[-500:])
                 return False
        else:
            print(f"❌ No logs found for today ({today})")
            return False
            
    finally:
        print("🛑 Killing server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)

if __name__ == "__main__":
    if verify():
        print("✨ Verification SUCCESS!")
        sys.exit(0)
    else:
        print("💀 Verification FAILED")
        sys.exit(1)
