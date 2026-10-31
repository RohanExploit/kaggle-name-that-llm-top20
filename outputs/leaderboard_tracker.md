# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: 2026-10-31 15:43:00 UTC

## Current Official Standing
- **Team**: `ROHAN Exploit`
- **Score**: **`0.6590`** (Submissions `57796150` & `57824131`)
- **Rank**: **`355`**
- **Today's Quota (2026-10-31 UTC)**: 2 used, **3 remaining**
- **Distance to Top 20**: **`+0.0180`**

## Target Cutoffs
- **Rank <= 20 Cutoff**: **`0.6770`** (`Nishant Dahal` Rank 20)
- **Top 10 Cutoff**: `0.6920`
- **Top 5 Cutoff**: `0.7080`
- **Rank 1**: `0.7320` (`Hod Carriers`)

## ACTIVE LIVE SUBMISSION (October 31, 2026 UTC)
- **Submission ID**: `57843309`
- **Kernel**: [`rohanexploit/name-that-llm-sota-v15`](https://www.kaggle.com/code/rohanexploit/name-that-llm-sota-v15) (version 2)
- **Status**: `SubmissionStatus.PENDING` (Running on Kaggle Cloud Dual T4 on hidden test set)
- **Description**: `SOTA v15 Master Ensemble: Damped Soft Sinkhorn + OOF Sibling Residual (Target >= 0.6770 Top 20)`
- **Architectural Enhancements**:
  - Full 6-Backbone V5 LoRA24 Dual-T4 Ensemble (Qwen3-14B/8B, Llama3.1-8B, Qwen2.5-14B, DeBERTa-v3-L)
  - CompStack 15% + ATLAS 20% Likelihood Beam Search + Qwen3.8-27B GSQ3
  - Roman's 5-Fold GroupKFold OOF Sibling Stylometry Residual (`sibling_style.py`) with 163 typography signals
  - Damped Soft Sinkhorn marginal balancing (`tau=0.45`, `target_mix=0.35`)
  - Target: Surpass `0.6770` and enter Top 20.

## Queued Candidate
- **Kernel**: [`rohanexploit/name-that-llm-summit-v16`](https://www.kaggle.com/code/rohanexploit/name-that-llm-summit-v16) (Summit Pinnacle 0.670+ base + OOF Sibling Residual)

## Previous Submissions
1. `57824131`: **`0.659`** (v14 Multi-LLM Ensemble + OOF Sibling Residual)
2. `57796150`: **`0.659`** (v13 Multi-LLM Ensemble V5 LoRA24 + ATLAS 20% + GSQ3)
3. `57794566`: **`0.340`** (DeBERTa Fold 2 + Stylometrics C=2.5)
4. `57791861`: **`0.336`** (55% DeBERTa + 45% Stylometrics C=4.0)
5. `57790968`: **`0.340`** (Restored Champion: Pure Fold 2 DeBERTa)
