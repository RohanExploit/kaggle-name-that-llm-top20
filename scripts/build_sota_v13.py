import os
import json

src_dir = r"E:\projects\kaggle\scratch\public_sota\light642"
dst_dir = r"E:\projects\kaggle\kernels\name-that-llm-ensemble"

nb = json.load(open(os.path.join(src_dir, "0-642-qwen3-8-27b-atlas.ipynb"), encoding="utf-8"))

resolver_code = """
# Dynamic Kaggle mount path resolver: handles both /kaggle/input/<slug>/... and /kaggle/input/datasets/<user>/<slug>/...
def _resolve_kaggle_path(p):
    if not isinstance(p, str) or not p.startswith('/kaggle/input/'):
        return p
    if Path(p).exists():
        return p
    parts = p.split('/')
    # Case 1: /kaggle/input/datasets/<user>/<slug>/... or /kaggle/input/notebooks/<user>/<slug>/...
    if len(parts) >= 6 and parts[3] in ('datasets', 'notebooks'):
        cand = '/kaggle/input/' + '/'.join(parts[5:])
        if Path(cand).exists():
            return cand
    # Case 2: /kaggle/input/competitions/<slug>/...
    if len(parts) >= 5 and parts[3] == 'competitions':
        cand = '/kaggle/input/' + '/'.join(parts[4:])
        if Path(cand).exists():
            return cand
    # Case 3: /kaggle/input/models/<user>/<model>/<framework>/<variation>/<ver>/...
    if len(parts) >= 8 and parts[3] == 'models':
        for cand in [
            '/kaggle/input/' + '/'.join(parts[5:]),
            '/kaggle/input/' + parts[5] + '/' + '/'.join(parts[6:]).lower(),
            '/kaggle/input/' + parts[5] + '/Transformers/' + '/'.join(parts[7:]),
            '/kaggle/input/' + parts[5] + '/transformers/' + '/'.join(parts[7:]),
        ]:
            if Path(cand).exists():
                return cand
    # Case 4: /kaggle/input/<slug>/... -> check /kaggle/input/datasets/*/<slug>/...
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

# Update Cell 43 ULTRA_ROOT
c43_src = "".join(nb["cells"][43]["source"])
c43_src = c43_src.replace(
    "ULTRA_ROOT = Path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl')",
    "ULTRA_ROOT = Path(_resolve_kaggle_path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl'))"
)
nb["cells"][43]["source"] = [line + "\n" for line in c43_src.splitlines()]

# Update Cell 44 NUMERIC_CACHE_PARTS and add verification of all paths!
verify_code = """NUMERIC_CACHE_PARTS = _resolve_obj(NUMERIC_CACHE_PARTS)
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
        for _ad in _spec['adapters']:
            assert Path(_ad).exists(), f'Missing adapter path: {_ad}'
    else:
        assert Path(_spec['ckpt']).exists(), f'Missing ckpt path: {_spec["ckpt"]}'
for _part in NUMERIC_CACHE_PARTS['parts']:
    _p = Path(NUMERIC_CACHE_PARTS['input_root']) / _part['name']
    assert _p.is_file() and _p.stat().st_size == _part['bytes'], f'Missing cache part: {_p}'
print('ALL KAGGLE INPUT MOUNTS, MODELS, KERNELS, AND CACHE PARTS VERIFIED 100%!', flush=True)
"""

c44_src = "".join(nb["cells"][44]["source"])
c44_lines = c44_src.splitlines()
idx_insert = next(i for i, l in enumerate(c44_lines) if l.startswith("CACHE_LABELS =")) + 1
c44_new = "\n".join(c44_lines[:idx_insert]) + "\n" + verify_code + "\n" + "\n".join(c44_lines[idx_insert:]) + "\n"
nb["cells"][44]["source"] = [line + "\n" for line in c44_new.splitlines()]

json.dump(nb, open(os.path.join(dst_dir, "submission_notebook.ipynb"), "w", encoding="utf-8"), indent=1)
print("Updated submission_notebook.ipynb successfully!")

