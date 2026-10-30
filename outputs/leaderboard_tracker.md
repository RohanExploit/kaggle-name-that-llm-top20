# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: 2026-10-30 13:09:29 UTC

## Target
- **Rank <= 20** (Current Cutoff: **0.6770** by Nishant Dahal)
- **Top 10 Cutoff**: 0.6920
- **Top 5 Cutoff**: 0.7080
- **Rank 1**: 0.7320 (Hod Carriers)

## Active Submission Status
- **Submission ID**: `57796150`
- **Kernel**: `rohanexploit/name-that-llm-ensemble-sub` (v13)
- **Status**: `SubmissionStatus.PENDING`
- **Score**: `Evaluating on hidden test split`

## Staged Next-Gen Champion
- **Kernel**: `rohanexploit/name-that-llm-sota-v14` (Verified COMPLETE on GPU)
- **GitHub**: https://github.com/RohanExploit/kaggle-name-that-llm-top20
- **Key Upgrade**:
  - 6-fold V5 LoRA24 multi-LLM ensemble + CompStack + ATLAS 20% + Qwen3.8-27B GSQ3
  - Roman's 5-Fold GroupKFold OOF-calibrated Sibling Stylometry Residual (`sibling_style.py`)
  - Within-family voice distinction (Claude, Gemini, GPT, GLM)
  - 100% exact cross-family log-sum-exp probability conservation
