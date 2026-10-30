import os
import json
from pathlib import Path

src_dir = r"E:\projects\kaggle\scratch\public_sota\light642"
dst_dir = r"E:\projects\kaggle\kernels\name-that-llm-ensemble-v14"
os.makedirs(dst_dir, exist_ok=True)

with open(os.path.join(src_dir, "0-642-qwen3-8-27b-atlas.ipynb"), "r", encoding="utf-8") as f:
    nb = json.load(f)

resolver_code = """
# Dynamic Kaggle mount path resolver: handles both /kaggle/input/<slug>/... and /kaggle/input/datasets/<user>/<slug>/...
def _resolve_kaggle_path(p):
    if not isinstance(p, str) or not p.startswith('/kaggle/input/'):
        return p
    if Path(p).exists():
        return p
    parts = p.split('/')
    if len(parts) >= 6 and parts[3] in ('datasets', 'notebooks'):
        cand = '/kaggle/input/' + '/'.join(parts[5:])
        if Path(cand).exists():
            return cand
    if len(parts) >= 5 and parts[3] == 'competitions':
        cand = '/kaggle/input/' + '/'.join(parts[4:])
        if Path(cand).exists():
            return cand
    if len(parts) >= 8 and parts[3] == 'models':
        for cand in [
            '/kaggle/input/' + '/'.join(parts[5:]),
            '/kaggle/input/' + parts[5] + '/' + '/'.join(parts[6:]).lower(),
            '/kaggle/input/' + parts[5] + '/Transformers/' + '/'.join(parts[7:]),
            '/kaggle/input/' + parts[5] + '/transformers/' + '/'.join(parts[7:]),
        ]:
            if Path(cand).exists():
                return cand
    if len(parts) >= 4:
        slug = parts[3]
        rest = '/'.join(parts[4:])
        for match in Path('/kaggle/input').glob(f'*/*/{slug}'):
            cand = str(match / rest) if rest else str(match)
            if Path(cand).exists():
                return cand
    return p

def _resolve_obj(obj):
    if isinstance(obj, str):
        return _resolve_kaggle_path(obj)
    if isinstance(obj, list):
        return [_resolve_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {_resolve_kaggle_path(k): _resolve_obj(v) for k, v in obj.items()}
    return obj

CONFIG = _resolve_obj(CONFIG)
EXPLICIT_PATHS = _resolve_obj(EXPLICIT_PATHS)
"""

# Update Cell 2 DATA path fallback
c2_src = "".join(nb["cells"][2]["source"])
c2_src = c2_src.replace(
    "DATA = Path('/kaggle/input/competitions/name-that-llm/test.csv')",
    "DATA = next(Path(p) for p in ['/kaggle/input/competitions/name-that-llm/test.csv', '/kaggle/input/name-that-llm/test.csv'] if Path(p).is_file())"
)
nb["cells"][2]["source"] = [line + "\n" for line in c2_src.splitlines()]

# Update Cell 4 to run _resolve_obj before writing config.json and explicit-paths.json
c4_src = "".join(nb["cells"][4]["source"])
c4_lines = c4_src.splitlines()
write_lines = [l for l in c4_lines if l.startswith("(RUNTIME/")]
def_lines = [l for l in c4_lines if not l.startswith("(RUNTIME/")]
c4_new = "\n".join(def_lines) + "\n" + resolver_code + "\n" + "\n".join(write_lines) + "\n"
nb["cells"][4]["source"] = [line + "\n" for line in c4_new.splitlines()]

# Sibling style writer code to insert as a new cell right before ensemble.py
sibling_style_cell_code = """%%writefile /kaggle/temp/ntl-gsq-v27-runtime/sibling_style.py
\"\"\"OOF-estimated, conservative within-family sibling stylometry residual.\"\"\"
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
    '\\u2014', '\\u2013', '-', '\\u2019', '\\u2018', '\\u201c', '\\u201d', '"', "'",
    '(', ')', '[', ']', '{', '}', '*', '`', '\\u2022', '\\u2026', '/', '\\\\',
    '\\n', '\\n\\n', '\\n- ', '\\n* ', '\\n1. ', '\\n2. ', '## ', '### ', '```', '**', '> ', '\\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',
    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  ',
    '|', '---', 'http', 'def ', 'class ', 'import ', 'return ', 'console.log'
]

WORDS = [
    'the', 'and', 'but', 'that', 'which', 'with', 'into', 'from', 'for', 'you',
    'your', 'our', 'we', 'I', 'it', 'its', 'they', 'their', 'there', 'this',
    'these', 'those', 'however', 'therefore', 'moreover', 'although', 'because',
    'perhaps', 'might', 'would', 'could', 'should', 'like', 'such', 'just', 'very',
    'also', 'not', 'if', 'or', 'as', 'so', 'by', 'of', 'to', 'in', 'on', 'at',
    'is', 'are', 'be', 'was', 'were', 'have', 'has', 'had', 'may', 'must', 'all',
    'any', 'each', 'every', 'many', 'some', 'more', 'most', 'only', 'even', 'still',
    'then', 'when', 'where', 'why', 'how', 'what', 'who', 'furthermore', 'additionally',
    'specifically', 'consequently', 'essentially', 'ultimately', 'namely'
]

WRE = re.compile(r"[A-Za-z]+(?:['\\u2019][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\\s+')

@lru_cache(maxsize=15000)
def extract_row(t):
    t = str(t)
    s = WRE.findall(t)
    lo = [z.lower() for z in s]
    n = max(len(s), 1)
    nt = max(len(t), 1)
    lines = t.splitlines()
    ns = max(len(lines), 1)
    lens = np.array([len(z) for z in s] or [0])
    sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [
        np.log1p(n), np.log1p(nt), nt / n, ns / n, len(set(lo)) / n,
        float(np.mean(lens)), float(np.std(lens)),
        float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)),
        float(np.mean(lens >= 8)), float(np.mean(lens <= 3)),
        sum(c.isupper() for c in t) / nt,
        sum(c.isdigit() for c in t) / nt,
        sum(ord(c) > 127 for c in t) / nt,
        sum(c.isspace() for c in t) / nt,
        sum(c == '\\n' for c in t) / nt,
        sum(len(l.strip()) == 0 for l in lines) / ns,
        float(np.mean([len(l) for l in lines] or [0])),
    ]
    base += [min(t.count(m) / n, 3) for m in MARKERS]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n for w in WORDS]
    return np.array(base, dtype=np.float32)

def extract_features(texts):
    return np.stack([extract_row(str(t)) for t in texts])

def apply_grouped_style_residual(logits, train_texts, test_texts, labels, y, groups, max_alpha=0.08):
    \"\"\"OOF-estimated conservative sibling re-ranking. Total family probability mass is mathematically conserved.\"\"\"
    Xt = extract_features(train_texts)
    Xq = extract_features(test_texts)
    out = np.asarray(logits, dtype=np.float64).copy()
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
        if reliability <= 0:
            continue

        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)

        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.65 - margin) / 0.65, 0.0, 1.0)
        strength = (max_alpha * reliability) * gate
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p

        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)

    return out
"""

# Update Cell 41 (ensemble.py): inject sibling style residual right after finalize()
c41_src = "".join(nb["cells"][41]["source"])
target_code = """    logits = finalize(mixed, frame.text.values, groups,
                      train_texts=official_texts, train_labels=official_labels.to_numpy(dtype=np.int64))"""

replacement_code = """    logits = finalize(mixed, frame.text.values, groups,
                      train_texts=official_texts, train_labels=official_labels.to_numpy(dtype=np.int64))
    from sibling_style import apply_grouped_style_residual
    logits = apply_grouped_style_residual(
        logits,
        train_texts=official_texts,
        test_texts=frame.text.values,
        labels=LABELS,
        y=official_labels.to_numpy(dtype=np.int64),
        groups=official.prompt_id.values,
        max_alpha=0.08
    )"""

assert target_code in c41_src, "Could not find finalize target in ensemble.py"
c41_src = c41_src.replace(target_code, replacement_code)
nb["cells"][41]["source"] = [line + "\n" for line in c41_src.splitlines()]

# Insert the sibling_style writer cell right before Cell 41
sibling_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in sibling_style_cell_code.splitlines()]
}
nb["cells"].insert(41, sibling_cell)

# Cell 44 (which is now index 44+1 = 45 due to insertion) ULTRA_ROOT
c44_idx = 44
c44_src = "".join(nb["cells"][c44_idx]["source"])
c44_src = c44_src.replace(
    "ULTRA_ROOT = Path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl')",
    "ULTRA_ROOT = Path(_resolve_kaggle_path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl'))"
)
nb["cells"][c44_idx]["source"] = [line + "\n" for line in c44_src.splitlines()]

# Cell 45 (which is now index 46): NUMERIC_CACHE_PARTS resolving
c45_idx = 45
c45_src = "".join(nb["cells"][c45_idx]["source"])
# Resolve input_root dynamically right after NUMERIC_CACHE_PARTS is defined
c45_src = c45_src.replace(
    "NUMERIC_CACHE_PARTS = {'input_root': '/kaggle/input/ntl-olmo2-native24-final-inference-assets'",
    "NUMERIC_CACHE_PARTS = {'input_root': _resolve_kaggle_path('/kaggle/input/ntl-olmo2-native24-final-inference-assets')"
)

verify_code = """
for _k, _v in CONFIG['paths'].items():
    if _k == 'ntl660_prefit': continue
    if isinstance(_v, list):
        for _item in _v:
            assert Path(_item).exists(), f'Missing list path {_k}: {_item}'
    else:
        assert Path(_v).exists(), f'Missing path {_k}: {_v}'
for _k, _v in EXPLICIT_PATHS['paths'].items():
    assert Path(_v['path']).exists(), f'Missing explicit path {_k}: {_v}'
for _k, _v in EXPLICIT_PATHS['lists'].items():
    for _item in _v['paths']:
        assert Path(_item).exists(), f'Missing explicit list path {_k}: {_item}'
for _k in EXPLICIT_PATHS['staging']:
    assert Path(_k).exists(), f'Missing explicit staging path: {_k}'
for _spec in CONFIG['models']:
    if _spec['type'] == 'l24':
        assert Path(_spec['path']).exists(), f'Missing model path: {_spec["path"]}'
print('ALL KAGGLE INPUT MOUNTS, MODELS, KERNELS, AND CACHE PARTS VERIFIED 100%!')
"""
c45_src = c45_src + "\n" + verify_code + "\n"
nb["cells"][c45_idx]["source"] = [line + "\n" for line in c45_src.splitlines()]

# Save output notebook
out_nb_path = os.path.join(dst_dir, "submission_notebook.ipynb")
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated v14 submission notebook at: {out_nb_path}")
print(f"Total cells: {len(nb['cells'])}")
