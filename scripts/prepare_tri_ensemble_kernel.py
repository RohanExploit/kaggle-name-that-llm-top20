import os, json, shutil

os.makedirs('kernels/name-that-llm-tri-ensemble', exist_ok=True)

# 1. Copy notebook
src_nb = 'notebooks/04_tri_ensemble_submission_kernel.ipynb'
dst_nb = 'kernels/name-that-llm-tri-ensemble/submission_notebook.ipynb'
shutil.copyfile(src_nb, dst_nb)
print(f"Copied {src_nb} -> {dst_nb}")

# 2. Write kernel metadata
meta = {
    "id": "rohanexploit/name-that-llm-tri-ensemble",
    "title": "name-that-llm-tri-ensemble",
    "code_file": "submission_notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": ["rohanexploit/ntl-deberta-high-score-model"],
    "kernel_sources": [],
    "competition_sources": ["name-that-llm"]
}

with open('kernels/name-that-llm-tri-ensemble/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Created kernels/name-that-llm-tri-ensemble/kernel-metadata.json successfully!")
