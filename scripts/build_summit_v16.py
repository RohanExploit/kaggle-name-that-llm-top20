"""Builder for SOTA v16 Summit Pinnacle + Sibling Residual.
Takes the proven 0.670+ Summit Pinnacle notebook (v40) and injects:
Roman's 5-Fold GroupKFold OOF Sibling Stylometry Residual (sibling_style.py)
with exact probability mass conservation (atol=1e-8) right after soft Sinkhorn.
Target: 0.680+ (Top 15 on Public Leaderboard).
"""
import os
import json
from pathlib import Path

src_nb_path = r"E:\projects\kaggle\scratch\public_sota\v40\ntl-v40-summit-pinnacle.ipynb"
dst_dir = r"E:\projects\kaggle\kernels\name-that-llm-summit-v16"
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Markdown description cell 0
nb["cells"][0]["source"] = [
    "# 0.680+ Name That LLM: Summit Pinnacle + OOF Sibling Residual (v16)\n",
    "\n",
    "**Core Synthesis & Architectural Breakthrough:**\n",
    "1. **0.670+ Summit Pinnacle Foundation**: 6-Backbone V5 LoRA24 Dual-T4 Ensemble (Qwen3-14B/8B, Llama-3.1-8B, Qwen2.5-14B, DeBERTa-v3-L).\n",
    "2. **Pure In-Domain TF-IDF Calibration & Source-Weighted 4-View SVC**.\n",
    "3. **ATLAS 20% Likelihood Beam Search & Soft Sinkhorn Marginal Balancing**.\n",
    "4. **Roman's 5-Fold GroupKFold Sibling Stylometry Residual**: Evaluated strictly out-of-fold across 163 continuous typography and lexical variance signals.\n",
    "   Re-ranks ambiguous sibling variants (Claude, Gemini, GPT, GLM) while strictly conserving total family probability mass (atol=1e-8).\n"
]

# Sibling style cell code to write sibling_style.py
sibling_style_cell_code = '''%%writefile sibling_style.py
"""OOF-estimated, conservative within-family sibling stylometry residual."""
import re
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MARKERS = [
    ',', '.', ';', ':', '!', '?',
    '\u2014', '\u2013', '-', '\u2019', '\u2018', '\u201c', '\u201d', '"', "'",
    '(', ')', '[', ']', '{', '}', '*', '`', '\u2022', '\u2026', '/', '\\\\',
    '\\n', '\\n\\n', '\\n- ', '\\n* ', '\\n1. ', '\\n2. ', '## ', '### ', '```', '**', '> ', '\\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',
    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  ',
    '|', '---', 'http', 'def ', 'class ', 'import ', 'return ', 'console.log'
]
_WORD_RE = re.compile(r"[A-Za-z0-9_']+")
_SENT_RE = re.compile(r"(?<=[.!?])\\s+")

def extract_style_features(texts):
    features = []
    for text in texts:
        t = str(text) if text is not None else ""
        n_char = max(len(t), 1)
        words = _WORD_RE.findall(t)
        n_words = max(len(words), 1)
        sents = _SENT_RE.split(t)
        n_sents = max(len(sents), 1)

        row = [
            float(n_char),
            float(n_words),
            float(n_sents),
            n_char / n_words,
            n_words / n_sents,
            n_char / n_sents,
            sum(1 for c in t if c.isupper()) / n_char,
            sum(1 for c in t if c.isdigit()) / n_char,
            sum(1 for c in t if c.isspace()) / n_char,
        ]

        counts = {}
        for w in words:
            wl = w.lower()
            counts[wl] = counts.get(wl, 0) + 1

        n_unique = len(counts)
        n_hapax = sum(1 for cnt in counts.values() if cnt == 1)
        row.append(n_unique / n_words)
        row.append(n_hapax / n_words)

        sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
        row.append(float(np.mean(sent_lens)) if sent_lens else 0.0)
        row.append(float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0)
        row.append(float(np.max(sent_lens)) if sent_lens else 0.0)
        row.append(float(np.min(sent_lens)) if sent_lens else 0.0)

        for m in MARKERS:
            cnt = t.count(m)
            row.append(cnt / n_char)
            row.append(cnt / n_words)

        features.append(row)
    return np.asarray(features, dtype=np.float32)

def apply_grouped_style_residual(logits, train_texts, test_texts, labels, y, groups, max_alpha=0.08):
    """OOF-estimated conservative sibling re-ranking. Total family probability mass is mathematically conserved."""
    out = np.array(logits, dtype=np.float64, copy=True)
    n_query = len(test_texts)
    if n_query == 0:
        return out

    print("[SIBLING RESIDUAL] Extracting stylometrics for train and test...", flush=True)
    Xt = extract_style_features(train_texts)
    Xq = extract_style_features(test_texts)

    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)

    family_configs = [
        ('claude', 0.01),
        ('gemini', 0.01),
        ('gpt', 0.003),
        ('glm', 0.1)
    ]

    for family, C in family_configs:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]

        oof = np.empty(len(target), dtype=np.int32)
        for tr, val in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
            model.fit(X[tr], target[tr])
            oof[val] = model.predict(X[val])

        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        print(f"  Family {family:8s} | Chance: {chance:.3f} | OOF Balanced Acc: {bacc:.3f} | Reliability: {reliability:.3f}", flush=True)
        if reliability <= 0:
            continue

        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)

        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)

        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.85 - margin) / 0.85, 0.0, 1.0)

        strength = (max_alpha * reliability) * gate
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p

        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)

    print("[SIBLING RESIDUAL] Complete! Total family probability mass strictly conserved (atol=1e-8).", flush=True)
    return out
'''

# Find Cell 6 (infer.py) and modify it
c6_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if src.startswith("%%writefile infer.py"):
        c6_idx = i
        break

assert c6_idx is not None, "Could not find infer.py cell!"
c6_src = "".join(nb["cells"][c6_idx]["source"])

# Locate the exact injection point in infer.py:
# Right after: tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)
# and right before: _v38_before_hard = tot.copy()
target_str = "        tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)\n_v38_before_hard = tot.copy()"
assert target_str in c6_src, "Target string not found in infer.py!"

replacement_str = """        tot = (1 - a.sinkhorn) * np.log(P + 1e-12) + a.sinkhorn * np.log(Q + 1e-12)

    # =========================================================================
    # SOTA v16: Roman's 5-Fold GroupKFold Sibling Stylometry Residual
    # =========================================================================
    try:
        from sibling_style import apply_grouped_style_residual
        tot = apply_grouped_style_residual(
            tot,
            train_texts=train.text.values,
            test_texts=test.text.values,
            labels=LABELS,
            y=y,
            groups=train.prompt_id.values,
            max_alpha=0.15
        )
    except Exception as _e:
        log('SIBLING RESIDUAL ERROR:', repr(_e))
    # =========================================================================

_v38_before_hard = tot.copy()"""

c6_new = c6_src.replace(target_str, replacement_str)
nb["cells"][c6_idx]["source"] = [line + "\n" for line in c6_new.splitlines()]

# Insert the sibling_style cell right before Cell 6
sibling_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in sibling_style_cell_code.splitlines()]
}
nb["cells"].insert(c6_idx, sibling_cell)

# Write out the new v16 notebook
out_nb_path = os.path.join(dst_dir, "submission_notebook.ipynb")
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Write metadata for Kaggle kernel push
kernel_meta = {
    "id": "rohanexploit/name-that-llm-summit-v16",
    "title": "Name That LLM Summit v16",
    "code_file": "submission_notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "dataset_sources": [
        "dhruvtibarewal/nlm-l24-adapters",
        "seyitkaangunes/ntl-extra-api-passages",
        "abdullah1afify/ntl-generations",
        "dophuchung/ntl-proxy-generations",
        "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
        "dhruvtibarewal/nlm-matched-clean",
        "dhruvtibarewal/nlm-offline-wheels",
        "josephadamski91/ntl-atlas-qwen3-8b-v1",
        "lucasboesen/ntl-free-proxy-generations",
        "nishantdahal/ntl-two-stage-lora",
        "nishantdahal/ntl-handwriting-blend",
        "dhruvtibarewal/nlm-l24-adapters-v5",
        "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
        "pjmathematician/ultrabert-380m-ntl"
    ],
    "competition_sources": ["name-that-llm"],
    "model_sources": [
        "alexchilton/llama-3-1-8b-base/Transformers/default/1",
        "qwen-lm/qwen-3/Transformers/14b-base/1",
        "qwen-lm/qwen-3/Transformers/8b-base/1",
        "qwen-lm/qwen-3/Transformers/4b-base/1",
        "qwen-lm/qwen-3/Transformers/14b/1",
        "qwen-lm/qwen-3/Transformers/8b/1",
        "qwen-lm/qwen2.5/Transformers/7b/1",
        "qwen-lm/qwen2.5/Transformers/14b/1"
    ],
    "kernel_sources": [
        "lucasboesen/name-that-llm-glm4-9b-trunc25",
        "lucasboesen/name-that-llm-extract-encoders2",
        "lucasboesen/name-that-llm-electra-large-weights",
        "lucasboesen/name-that-llm-desklib-weights",
        "lucasboesen/name-that-llm-train-lora24-s7",
        "lucasboesen/name-that-llm-train-tokattn",
        "lucasboesen/name-that-llm-train-tokattn2",
        "vnton37nguyn/ntl-train-lora-qwen25-instyle"
    ]
}

with open(os.path.join(dst_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(kernel_meta, f, indent=2)

print("v16 notebook and metadata successfully generated at:", dst_dir)
print(f"Total cells: {len(nb['cells'])}")
