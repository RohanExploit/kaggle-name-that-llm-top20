# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: 2026-10-31 02:34:00 UTC

## Current Official Standing
- **Team**: `ROHAN Exploit`
- **Score**: **`0.6590`** (Submission `57796150`)
- **Rank**: **`355`** (Surged **+2,877 ranks** from 3,232)
- **Today's Quota (2026-10-31 UTC)**: 1 used, **4 remaining**
- **Distance to Top 20**: **`+0.0180`**

## Target Cutoffs
- **Rank <= 20 Cutoff**: **`0.6770`** (`Nishant Dahal` Rank 20, `Maniceet Sahay` Rank 21 at 0.6750)
- **Top 10 Cutoff**: `0.6920`
- **Top 5 Cutoff**: `0.7080`
- **Rank 1**: `0.7320` (`Hod Carriers`)

## ACTIVE LIVE SUBMISSION (October 31, 2026 UTC)
- **Submission ID**: `57824131`
- **Kernel**: [`rohanexploit/name-that-llm-sota-v14`](https://www.kaggle.com/code/rohanexploit/name-that-llm-sota-v14) (version 2)
- **Status**: `SubmissionStatus.PENDING` (Running on Kaggle Cloud Dual T4 on hidden test set)
- **Description**: `SOTA v14 Multi-LLM Ensemble + OOF Sibling Residual (Target >= 0.6770 Top 20)`
- **Architectural Breakthrough**:
  - 6-backbone V5 LoRA24 (Qwen3-14B/8B, Llama3.1-8B, Qwen2.5-14B, DeBERTa-v3-L)
  - CompStack 15% + ATLAS 20% + Qwen3.8-27B GSQ3 7.5%
  - Roman's 5-Fold GroupKFold OOF Sibling Stylometry Residual (`sibling_style.py`)
  - Within-family voice distinction (Claude, Gemini, GPT, GLM)
  - 100% exact cross-family log-sum-exp probability conservation
  - Directly targeting the `+0.0180` gap to enter Top 20 (`Score >= 0.6770`).

## Previous Submissions (2026-10-30 UTC)
1. `57796150`: **`0.659`** - Multi-LLM Ensemble V5 LoRA24 + CompStack + ATLAS 20% + Qwen3.8-27B GSQ3
2. `57794566`: **`0.340`** - Refined Forensics: DeBERTa Fold 2 + Stylometrics C=2.5
3. `57791861`: **`0.336`** - Tuned Forensics: 55% DeBERTa + 45% Stylometrics C=4.0
4. `57790968`: **`0.340`** - Restored Champion: Pure Fold 2 DeBERTa + Stylometrics
5. `57790374`: **`0.267`** - Tri-Ensemble: 55% DeBERTa Soup + 35% Stylometry + 10% LightGBM
