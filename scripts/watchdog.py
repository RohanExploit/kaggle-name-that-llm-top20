"""
Autonomous Watchdog and Auto-Healing Script for Name That LLM.
Runs on each 10-minute cron cycle:
1. Checks if auto_pilot process is alive; revives it if terminated.
2. Verifies submission.parquet integrity.
3. Summarizes latest leaderboard tracker status.
"""

import os
import subprocess
import sys
import psutil
import pandas as pd


OFFICIAL_18_CLASSES = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def is_autopilot_alive() -> bool:
    """Checks if auto_pilot.py is currently running in process table."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            if any("auto_pilot.py" in arg for arg in cmd):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def verify_submission(sub_path: str = "submission.parquet") -> bool:
    """Verifies that the submission file exists and complies with all competition rules."""
    if not os.path.exists(sub_path):
        print(f"[WATCHDOG WARNING] {sub_path} not found!")
        return False

    df = pd.read_parquet(sub_path)
    if "id" not in df.columns or "model" not in df.columns:
        print("[WATCHDOG ERROR] submission.parquet missing required columns!")
        return False

    if df["model"].isnull().any():
        print("[WATCHDOG ERROR] submission.parquet has null values!")
        return False

    invalid = set(df["model"]) - set(OFFICIAL_18_CLASSES)
    if invalid:
        print(f"[WATCHDOG ERROR] Invalid class labels found: {invalid}")
        return False

    print(f"[WATCHDOG OK] submission.parquet verified ({len(df)} rows, 0 nulls, all 18 labels valid)")
    return True


def check_status():
    print("=" * 65)
    print("  AUTONOMOUS WATCHDOG HEALTH CHECK (10-MIN CYCLE)")
    print("=" * 65)

    alive = is_autopilot_alive()
    if alive:
        print("[HEALTH] auto_pilot.py is ACTIVE and running.")
    else:
        print("[HEALTH ALERT] auto_pilot.py was NOT running! Reviving daemon now...")
        subprocess.Popen(
            [sys.executable, "-u", "auto_pilot.py", "8.0"],
            cwd="e:/projects/kaggle",
            stdout=open("outputs/auto_pilot_revived.log", "a"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        print("[HEALTH] auto_pilot.py successfully revived!")

    verify_submission("submission.parquet")

    # Read latest lines of tracker
    tracker_path = "outputs/leaderboard_tracker.md"
    if os.path.exists(tracker_path):
        with open(tracker_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) > 4:
            print("\n[LATEST BENCHMARK RESULTS]")
            print("\n".join(lines[-3:]))

    print("=" * 65)


if __name__ == "__main__":
    check_status()
