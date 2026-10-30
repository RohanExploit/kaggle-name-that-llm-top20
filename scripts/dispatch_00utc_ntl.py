"""Autonomous 00:00 UTC Quota Unlock & Top-20 Leaderboard Tracker for Kaggle Name That LLM.

Goal: Climb into Top 20 (Target Score >= 0.6770, current cutoff held by Nishant Dahal at 0.6770).
Strategy:
  1. Primary Thrust: SOTA v14 Multi-LLM Ensemble + Roman's 5-Fold GroupKFold OOF Sibling Residual
     (rohanexploit/name-that-llm-sota-v14 v2) - verified COMPLETE on Dual T4 Cloud GPU.
  2. Live Monitoring: Poll submission status until COMPLETE, fetch public score, download latest LB.
  3. Git Sync: Commit and push score, rank, and metrics to https://github.com/RohanExploit/kaggle-name-that-llm-top20
"""
import datetime
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\projects\kaggle")
COMPETITION = "name-that-llm"
OUTPUTS = ROOT / "outputs"
TRACKER_PATH = OUTPUTS / "leaderboard_tracker.md"

TRACKS = [
    {
        "name": "Track 1 (Champion): SOTA v14 Multi-LLM Ensemble + OOF Sibling Residual",
        "kernel": "rohanexploit/name-that-llm-sota-v14",
        "version": 2,
        "file": "submission.parquet",
        "message": "SOTA v14 Multi-LLM Ensemble + OOF Sibling Residual (Target >= 0.6770 Top 20)",
    }
]


def seconds_until_midnight_utc():
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=5, microsecond=0
    )
    diff = (midnight - now).total_seconds()
    return max(0.0, diff)


def submit_track(track):
    print(f"\n[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Submitting: {track['name']}")
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-k", track["kernel"],
        "-v", str(track["version"]),
        "-f", track["file"],
        "-m", track["message"]
    ]
    print("Command:", " ".join(cmd))
    res = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    return res.returncode == 0


def poll_latest_submission(timeout_sec=7200):
    start = time.time()
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Polling latest submission status...")
    while time.time() - start < timeout_sec:
        res = subprocess.run(
            f"kaggle competitions submissions -c {COMPETITION} --csv",
            shell=True, capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = [l for l in res.stdout.splitlines() if not l.startswith("Warning")]
            if len(lines) > 1:
                import io
                df = pd.read_csv(io.StringIO("\n".join(lines)))
                latest = df.iloc[0]
                status = str(latest.get("status", ""))
                score = latest.get("publicScore", None)
                print(f"  Ref: {latest.get('ref')} | Status: {status} | Score: {score}")
                if "COMPLETE" in status:
                    return float(score) if pd.notna(score) else None
                if "ERROR" in status:
                    print("  Submission resulted in ERROR!")
                    return None
        time.sleep(30)
    print("Polling timed out!")
    return None


def fetch_and_evaluate_leaderboard(score):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] Fetching fresh leaderboard...")
    zip_path = OUTPUTS / "name-that-llm.zip"
    subprocess.run(
        f"kaggle competitions leaderboard -c {COMPETITION} --download -p {OUTPUTS}",
        shell=True, check=True
    )
    extract_dir = OUTPUTS / "latest_lb"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    csvs = list(extract_dir.glob("*.csv"))
    if not csvs:
        return
    df = pd.read_csv(csvs[0])
    my_row = df[df["TeamMemberUserNames"].str.contains("rohanexploit", na=False, case=False)]
    if not my_row.empty:
        rank = int(my_row.iloc[0]["Rank"])
        team = str(my_row.iloc[0]["TeamName"])
        curr_score = float(my_row.iloc[0]["Score"])
        top20_score = float(df.iloc[19]["Score"]) if len(df) >= 20 else 0.6770
        print(f"*** CURRENT STANDING: Rank {rank} | Team: {team} | Score: {curr_score} ***")
        print(f"*** TOP 20 CUTOFF: Rank 20 = {top20_score} (Gap: {top20_score - curr_score:+.4f}) ***")
        
        # Update tracker
        utc_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        tracker_content = f"""# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: {utc_str}

## Current Official Standing
- **Team**: `{team}`
- **Score**: **`{curr_score:.4f}`**
- **Rank**: **`{rank}`**
- **Top 20 Cutoff**: **`{top20_score:.4f}`**
- **Distance to Top 20**: **`{top20_score - curr_score:+.4f}`**

## Goal Status
- Target: `Rank <= 20`
- Result: {"TOP 20 ACHIEVED! SUCCESS!" if rank <= 20 else "In Progress towards Top 20"}
"""
        TRACKER_PATH.write_text(tracker_content, encoding="utf-8")
        subprocess.run("git add outputs/leaderboard_tracker.md", cwd=str(ROOT), shell=True)
        subprocess.run(f'git commit -m "docs: sync standing Rank {rank} ({curr_score})"', cwd=str(ROOT), shell=True)
        subprocess.run("git push origin main", cwd=str(ROOT), shell=True)
        
        if rank <= 20:
            print("\n" + "="*60)
            print("GOAL ACHIEVED: RANK <= 20 VERIFIED ON OFFICIAL LEADERBOARD!")
            print("="*60 + "\n")


def main():
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] NTL 00:00 UTC Dispatcher Initialized.")
    secs = seconds_until_midnight_utc()
    hours = secs / 3600.0
    print(f"Time until 00:00:05 UTC reset: {secs:.1f}s ({hours:.2f} hours)")
    
    if secs > 10:
        print(f"Sleeping until 00:00:05 UTC (consuming 0% CPU, 0% GPU)...")
        time.sleep(secs)
        
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] QUOTA UNLOCKED! Launching Track 1...")
    track = TRACKS[0]
    ok = submit_track(track)
    if ok:
        score = poll_latest_submission()
        if score is not None:
            fetch_and_evaluate_leaderboard(score)
    else:
        print("Submission command returned non-zero code. Retrying after 15s...")
        time.sleep(15)
        submit_track(track)


if __name__ == "__main__":
    main()
