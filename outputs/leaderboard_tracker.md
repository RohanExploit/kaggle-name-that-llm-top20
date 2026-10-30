# Kaggle Name That LLM - Live Leaderboard Tracker

Last Updated: 2026-10-30 17:25:00 UTC

## Current Official Standing
- **Team**: `ROHAN Exploit`
- **Score**: **`0.6590`** (Submission `57796150`)
- **Rank**: **`355`** (Surged **+2,877 ranks** from 3,232)
- **Submissions Used**: 10 total (Today: 5/5)
- **Distance to Top 20**: **`+0.0180`**

## Target Cutoffs
- **Rank <= 20 Cutoff**: **`0.6770`** (`Nishant Dahal` Rank 20, `Maniceet Sahay` Rank 21 at 0.6750)
- **Top 10 Cutoff**: `0.6920`
- **Top 5 Cutoff**: `0.7080`
- **Rank 1**: `0.7320` (`Hod Carriers`)

## Historical Submissions Today (2026-10-30 UTC)
1. `57796150`: **`0.659`** - Multi-LLM Ensemble V5 LoRA24 + CompStack + ATLAS 20% + Qwen3.8-27B GSQ3
2. `57794566`: **`0.340`** - Refined Forensics: DeBERTa Fold 2 + Stylometrics C=2.5
3. `57791861`: **`0.336`** - Tuned Forensics: 55% DeBERTa + 45% Stylometrics C=4.0
4. `57790968`: **`0.340`** - Restored Champion: Pure Fold 2 DeBERTa + Stylometrics
5. `57790374`: **`0.267`** - Tri-Ensemble: 55% DeBERTa Soup + 35% Stylometry + 10% LightGBM

## Staged Next-Gen Champion (Ready for 00:00 UTC Reset)
- **Kernel**: [`rohanexploit/name-that-llm-sota-v14`](https://www.kaggle.com/code/rohanexploit/name-that-llm-sota-v14)
- **Status**: **`KernelWorkerStatus.COMPLETE`** (Verified green on Kaggle Cloud Dual T4)
- **GitHub**: [`https://github.com/RohanExploit/kaggle-name-that-llm-top20`](https://github.com/RohanExploit/kaggle-name-that-llm-top20)
- **Architectural Breakthrough in `v14`**:
  - Incorporates Roman's 5-Fold GroupKFold OOF-calibrated Sibling Stylometry Residual (`sibling_style.py`).
  - Distinguishes sibling variants within Claude, Gemini, GPT, and GLM families using 163-feature typography & lexical density vectors.
  - Strict mathematical conservation of total family probability mass (`logsumexp` match at `atol=1e-8`).
  - Directly targets the 0.018 gap to cross into Top 20 (`>= 0.6770`).
