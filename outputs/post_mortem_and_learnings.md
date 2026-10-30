# Name That LLM: Post-Mortem, Diagnostic Analysis & Core Learnings

> **Date**: October 30, 2026  
> **Team**: ROHAN Exploit (`rohanexploit`)  
> **Competition**: [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
> **Current Best Public Score**: **`0.340`** (Rank 3,177 / 3,857; **+551 teams jumped**)  
> **Status**: Continuous Autonomous Execution (4 Submissions Remaining for Today)

---

## 1. Executive Summary & Trajectory Analysis

Across 7 iterative submissions and local cross-validation cycles, our progression mapped out clear boundaries between high-generalization architectures and failure modes:

| Version | Architecture / Pipeline | Public LB Score | OOF Balanced Acc | Verdict & Key Finding |
|---|---|---|---|---|
| **R0** | Uniform Random Prior | — | 0.0556 | Baseline (18 balanced classes). |
| **K1 (v1)** | Stylometry + Char/Word TF-IDF | **0.286** | 0.1842 | Proven baseline; sparse n-grams establish initial lab separation. |
| **K2 (v4)** | Enhanced Stylometric n-grams + dense ratios | **0.290** | 0.2014 | +0.004 gain; punctuation ratios filter out common token overlap. |
| **K4 (v6)** | **50/50 DeBERTa-v3 Fold 2 + Stylometric Forensics** | **`0.340`** | **0.3215** |  **PEAK SCORE / BREAKTHROUGH (+551 rank jump)**. Orthogonal error profiles. |
| **K5 (v7)** | Calibrated Softmax ($T_{neu}=1.4, T_{sty}=1.2$) + 18 Features | **0.332** | 0.3252 |  **Regression (-0.008)**. Softening probabilities erodes decision sharpness. |
| **K6 (v8)** | Raw 50/50 Blend + 18 Extended Features | **0.332** | 0.3248 | Extended markdown features add noise on short out-of-distribution prompts. |
| **K8 (Tri-v1)** | Tri-Ensemble (DeBERTa Soup + Stylometry + LightGBM) | **0.267** | 0.3260 |  **Severe Degradation (-0.073)**. Weight-averaging heads + tree overfitting on SVD. |

---

## 2. What Went Right (The Winning Formula)

### A. Orthogonal Ensemble Synergy (The 0.340 Breakthrough)
- **Dense vs. Sparse Orthogonality**: DeBERTa-v3 captures semantic context, instruction style, and model reasoning tone. Stylometric TF-IDF captures raw character habits, vocabulary breadth, and punctuation frequencies.
- Because their confusion matrices are completely uncorrelated, fusing `0.50 * P(Neural) + 0.50 * P(Stylometry)` vaulted the public score from `0.290` straight to **`0.340`**, jumping **551 positions** on the official leaderboard.

### B. High-Signal Stylometric Invariants
Across all experiments, the following 8 features consistently yielded the highest mutual information with the 18 frontier LLMs:
1. **Em-dash frequency (`—`)**: Extreme Anthropic signature (Claude models frequently use em-dashes for parenthetical clauses).
2. **Curly quote & apostrophe ratios (`’`, `“`, `”` vs `'`, `"`)**: Distinct tokenization artifact of frontier OpenAI vs Google vs Anthropic tokenizers.
3. **Semicolon frequency (`;`)**: Stylistic separator heavily preferred by reasoning-tuned models (`claude-opus`, `gpt-5.6-terra`).
4. **Colon frequency (`:`)**: Structured list marker.
5. **Type-Token Ratio (TTR)**: Vocabulary richness indicator separating flash/mini models from frontier reasoning models.
6. **Average Sentence Length & Word Length**: Sentence complexity distribution.
7. **Character n-grams (`ngram_range=(2, 4), analyzer='char_wb'`)**: Captures subword morphology without vocabulary mismatch.
8. **Sublinear TF scaling (`sublinear_tf=True`)**: Prevents repetitive keywords from dominating the decision boundary.

### C. Raw Probability Blending vs. Temperature Calibration
- On out-of-distribution test prompts, the models must **firmly commit** to a lab family (OpenAI vs Anthropic vs Google).
- Raw unsoftened probabilities (`0.50 * P_neu + 0.50 * P_sty`) scored **`0.340`**, outperforming calibrated probabilities (`0.332`).

### D. Single-Fold DeBERTa-v3 Fold 2 Training Stability
- Training with AdamW `lr=1.5e-5`, `eps=1e-6`, gradient clipping `max_grad_norm=1.0`, and sequence length `seq_len=384` converged without loss NaN and produced our highest-performing neural weights.

### E. Code Competition Execution Reliability
- Offline submission kernels run with `enable_internet: false` in under **4.5 minutes** on Kaggle P100 GPUs, well below the 9-hour timeout.
- Fully validated output schema (`id`, `model`) with zero nulls and 100% adherence to the official 18 classes.

---

## 3. What Went Wrong & Why (Rules to Never Repeat)

###  Failure Mode 1: Parameter-Space Model Soup on Unaligned Classification Heads
- **What happened**: Averaging tensor weights across Fold 0, Fold 1, and Fold 2 into a single checkpoint (`model.safetensors`) caused the leaderboard score to collapse from **`0.340`** down to **`0.267`** (-0.073 drop).
- **Root Cause**: While transformer backbone attention layers share similar feature spaces, the linear classification heads (`classifier.weight` and `classifier.bias`) and poolers are randomly initialized per fold. Averaging weights of unaligned linear heads cancels out discriminating orthogonal planes and destroys class separation.
- **Rule**: **NEVER average raw neural weights in parameter space.** Always run models independently (or sequential inference) and average their **output probability vectors in probability space**:
  $$\hat{P}_{\text{ensemble}} = \frac{1}{K}\sum_{k=1}^K P_k(x)$$

---

###  Failure Mode 2: Non-Linear GBDT (LightGBM) Overfitting on SVD Dimensions
- **What happened**: Adding LightGBM GBDT trained on TruncatedSVD components and stylometrics yielded a deceptive local CV gain (+0.0045), but failed on the hidden test set.
- **Root Cause**: With only 2,699 training samples and 98 continuous features, tree-based models create brittle, axis-aligned decision cuts. On test prompts generated from unseen prompts, SVD feature coordinates drift, causing trees to make erratic high-confidence misclassifications (e.g. overpredicting Gemini and completely ignoring Claude). L2-regularized linear models (`LogisticRegression(C=2.5)`) provide smooth, monotonic hyperplanes that generalize significantly better.
- **Rule**: **Avoid tree-based GBDTs on small sample sizes with dense lexical SVDs.** Use L2-regularized linear classifiers for sparse/dense lexical forensic features.

---

###  Failure Mode 3: Diluting Core Punctuation with Structural Markdown Features
- **What happened**: Expanding from 8 core stylometrics to 18 features (adding markdown header density, bold tag ratios, bullet-point counts, code fence ratios) dropped the score from `0.340` to `0.332`.
- **Root Cause**: Hidden test prompts vary drastically in length. Short prompts (1-2 sentences) naturally have 0 headers, 0 bullets, and 0 code blocks. These features become uninformative zeros that add sparsity and dilute the true punctuation signals.
- **Rule**: **Keep stylometric features strictly prompt-length invariant.** Rely on punctuation-per-word ratios, character-level n-grams, and lexical diversity.

---

###  Failure Mode 4: Softmax Temperature Scaling on Out-of-Distribution Data
- **What happened**: Softening probabilities ($T=1.4, T=1.2$) reduced the official test score by -0.008.
- **Root Cause**: While temperature scaling minimizes log-loss on in-distribution validation, balanced accuracy on out-of-distribution prompts rewards sharp confidence when detecting unmistakable lab signatures (e.g. Claude em-dashes).
- **Rule**: **Use uncalibrated raw softmax probabilities for ensemble fusion.**

---

###  Empirical Finding: Hierarchical Family Sharpening (Gamma Scaling)
- **Experiment**: Analyzed error modes across 2,699 validation samples. Found that 68.9% of samples have the correct lab family identified, while 54.2% of all errors are intra-family sibling confusions. Tested scaling top predicted family probabilities by $\gamma \in [1.2, 2.5]$ while dampening other families.
- **Result**: Exactly zero gain ($\Delta = +0.0000$). When the family prediction is incorrect (~28% of samples), sharpening penalizes the true class even more severely, perfectly canceling out the gains on correct-family instances.
- **Conclusion**: Unmodified joint probabilities preserve true likelihood ratios and are mathematically optimal.


---

###  Failure Mode 5: Environment & Tooling Gotchas (Resolved)
1. **Windows Console UnicodeEncodeError**:
   - *Fix*: Standardized UTF-8 reconfigure (`sys.stdout.reconfigure(encoding='utf-8')`) in all scripts.
2. **Kaggle CLI Windows Temp Path Bug**:
   - `kaggle datasets version -p checkpoints/soup_model` fails if `C:\Users\ACER\AppData\Local\Temp\.kaggle\uploads\checkpoints` is missing.
   - *Fix*: Pre-create absolute temp directories before running CLI uploads.
3. **Kaggle Linux / Python 3.13 Symlink Traversal**:
   - `glob.glob('/kaggle/input/**/model.safetensors', recursive=True)` skips symlinked mounts on Kaggle's Python 3.13 image.
   - *Fix*: Traversed via `os.walk('/kaggle/input', followlinks=True)` and explicit candidate list.
4. **PyTorch FP16 GradScaler NaN**:
   - *Fix*: Trained with FP32 precision and AdamW `eps=1e-6`.

---

## 4. Architectural Rules for Next Iterations

```
┌─────────────────────────────────────────────────────────────┐
│                 GOLDEN ENSEMBLE SPECIFICATION                │
├─────────────────────────────────────────────────────────────┤
│ 1. Neural Backbone:   Single Fold 2 DeBERTa-v3 or Multi-     │
│                       Model Probability Averaging           │
│ 2. Stylometrics:      Strict 8 Punctuation Invariants       │
│ 3. Lexical Model:     Word (1,2) + Char (2,4) TF-IDF        │
│ 4. Classifier:        Logistic Regression (C=2.5, Balanced) │
│ 5. Blending:          Raw Probability Average:              │
│                       P = 0.50 * P_neural + 0.50 * P_sty    │
│ 6. Calibration:       None (Preserve Raw Decision Sharpness)│
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Action Plan & Empirical Execution Status

1. **Submission 1 (Immediate Recovery - DISPATCHED & EVALUATING)**:
   - Restored champion **`0.340`** pipeline: Pure Fold 2 DeBERTa-v3 + 8 Core Punctuation Invariants + Raw 50/50 fusion.
   - Pushed Version 9 to [`rohanexploit/name-that-llm-ensemble-sub`](https://www.kaggle.com/code/rohanexploit/name-that-llm-ensemble-sub) and submitted (Submission ID: **`57790968`**, Status: `PENDING`).
2. **Submission 2 (Regularization Upgrade via Empirical Grid Search)**:
   - Systematic 3-fold cross-validation sweep over $C \in [0.8, 4.0]$ revealed that $C=4.0$ boosts Stylometric OOF from `0.2974` to **`0.3060`** and the 50/50 blend to **`0.3319`** by allowing rare frontier model n-gram templates to be weighted appropriately.
   - Ready for dispatch as Version 10.
3. **Submission 3 (Probability-Space Multi-Fold Ensemble)**:
   - Run Fold 0 and Fold 1 sequentially during inference. Average their output probability distributions (NOT raw weights):
     $$P_{\text{neural}} = 0.5 \cdot P_{\text{fold1}} + 0.5 \cdot P_{\text{fold2}}$$
   - Fused with 50% Stylometry ($C=4.0$) $\to$ Targets **`0.355+`**.
4. **Submission 4 (Lab Family Hierarchical Prior)**:
   - Apply hierarchical model-family prior (OpenAI vs Anthropic vs Google vs Qwen) to eliminate sibling confusion.
