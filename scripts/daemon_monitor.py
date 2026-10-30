import subprocess
import time
import json
import os
import sys
from datetime import datetime

LOG_FILE = r"E:\projects\kaggle\outputs\monitor.log"
TRACKER_FILE = r"E:\projects\kaggle\outputs\leaderboard_tracker.md"

def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def check_submissions():
    try:
        res = subprocess.run(
            "kaggle competitions submissions -c name-that-llm --csv",
            shell=True, capture_output=True, text=True
        )
        if res.returncode != 0:
            log(f"Submissions check returned code {res.returncode}: {res.stderr.strip()[:120]}")
            return None
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.startswith("Warning:")]
        if len(lines) < 2:
            return None
        rows = []
        for l in lines[1:]:
            parts = l.split(",")
            if len(parts) >= 6:
                rows.append({
                    "ref": parts[0],
                    "fileName": parts[1],
                    "date": parts[2],
                    "status": parts[4],
                    "publicScore": parts[5] if len(parts) > 5 else ""
                })
        return rows
    except Exception as e:
        log(f"Exception checking submissions: {e}")
        return None

def update_tracker(active_sub, score):
    content = f"""# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Target
- **Rank <= 20** (Current Cutoff: **0.6770** by Nishant Dahal)
- **Top 10 Cutoff**: 0.6920
- **Top 5 Cutoff**: 0.7080
- **Rank 1**: 0.7320 (Hod Carriers)

## Active Submission Status
- **Submission ID**: `{active_sub.get('ref', '57796150')}`
- **Kernel**: `rohanexploit/name-that-llm-ensemble-sub` (v13)
- **Status**: `{active_sub.get('status', 'PENDING')}`
- **Score**: `{score or 'Evaluating on hidden test split'}`

## Staged Next-Gen Champion
- **Kernel**: `rohanexploit/name-that-llm-sota-v14` (Verified COMPLETE on GPU)
- **GitHub**: https://github.com/RohanExploit/kaggle-name-that-llm-top20
- **Key Upgrade**:
  - 6-fold V5 LoRA24 multi-LLM ensemble + CompStack + ATLAS 20% + Qwen3.8-27B GSQ3
  - Roman's 5-Fold GroupKFold OOF-calibrated Sibling Stylometry Residual (`sibling_style.py`)
  - Within-family voice distinction (Claude, Gemini, GPT, GLM)
  - 100% exact cross-family log-sum-exp probability conservation
"""
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    log("Starting Autonomous Kaggle Name-That-LLM Monitor Daemon v2...")
    last_status = None
    last_score = None
    
    while True:
        subs = check_submissions()
        if subs:
            top_sub = subs[0]
            status = top_sub["status"]
            score = top_sub["publicScore"]
            
            if status != last_status or score != last_score:
                log(f"Submission {top_sub['ref']}: Status={status}, Score={score}")
                last_status = status
                last_score = score
                update_tracker(top_sub, score)
                
                if status == "SubmissionStatus.COMPLETE":
                    log(f"SUBMISSION COMPLETED! Public Score: {score}")
                    break
        time.sleep(60)

if __name__ == '__main__':
    main()
