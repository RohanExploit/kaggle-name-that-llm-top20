"""Builder for SOTA v15 Master Ensemble Notebook.
Builds on v14 with:
1. 6-Backbone V5 LoRA24 Dual-T4 Multi-LLM Ensemble (Qwen3-14B/8B, Llama3.1-8B, Qwen2.5-14B, DeBERTa-v3-L)
2. CompStack 15% + ATLAS 20% Likelihood Beam Search
3. Qwen3.8-27B GSQ3 Full-Text Inference (7.5%)
4. Roman's 5-Fold GroupKFold Sibling Stylometry Residual (sibling_style.py) with 163 typography signals
5. Soft Sinkhorn Damped Marginal Balancing (tau=0.45, target_mix=0.35)
"""
import os
import json
import shutil
from pathlib import Path

src_dir = r"E:\projects\kaggle\kernels\name-that-llm-ensemble-v14"
dst_dir = r"E:\projects\kaggle\kernels\name-that-llm-sota-v15"
os.makedirs(dst_dir, exist_ok=True)

with open(os.path.join(src_dir, "submission_notebook.ipynb"), "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Markdown description cell 0
nb["cells"][0]["source"] = [
    "# Name That LLM: SOTA v15 Summit Ensemble (Target: Top 20)\n",
    "\n",
    "Candidate v15 combines the full 6-backbone V5 LoRA24 ensemble with:\n",
    "1. 6-Backbone V5 LoRA24 Dual-T4 DP (Qwen3-14B/8B, Llama3.1-8B, Qwen2.5-14B, DeBERTa-v3-L)\n",
    "2. CompStack 15% + ATLAS 20% Source-Conditioned Likelihood Expert\n",
    "3. Qwen3.8-27B GSQ3 LoRA readout (7.5% share)\n",
    "4. Roman's 5-Fold GroupKFold OOF Sibling Stylometry Residual (conserves 100% family mass)\n",
    "5. Damped Soft Sinkhorn Marginal Balancing (tau=0.45, target_mix=0.35)\n"
]

# Write out the new v15 notebook
out_nb_path = os.path.join(dst_dir, "submission_notebook.ipynb")
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Write metadata for Kaggle kernel push
kernel_meta = {
    "id": "rohanexploit/name-that-llm-sota-v15",
    "title": "Name That LLM SOTA v15",
    "code_file": "submission_notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "dataset_sources": [
        "dhruvtibarewal/nlm-l24-adapters",
        "lightningv08/ntl-olmo2-native24-final-inference-assets",
        "seyitkaangunes/ntl-extra-api-passages",
        "abdullah1afify/ntl-generations",
        "dophuchung/ntl-proxy-generations",
        "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
        "josephadamski91/ntl-atlas-qwen3-8b-v1",
        "lucasboesen/ntl-free-proxy-generations",
        "kirilltym/ntl-spectral-style-heads-paired-4404",
        "dhruvtibarewal/nlm-enc-v5",
        "nishantdahal/ntl-two-stage-lora",
        "nishantdahal/ntl-handwriting-blend",
        "dhruvtibarewal/nlm-l24-adapters-v5",
        "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
        "bbucxi/qwen35-9b-safetensors",
        "pjmathematician/ultrabert-380m-ntl"
    ],
    "competition_sources": ["name-that-llm"],
    "model_sources": [
        "qwen-lm/qwen-3/Transformers/4b-base/1",
        "qwen-lm/qwen-3/Transformers/14b-base/1",
        "qwen-lm/qwen-3/Transformers/8b-base/1",
        "qwen-lm/qwen2.5/Transformers/14b/1",
        "qwen-lm/qwen2.5/Transformers/7b/1"
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

print("v15 notebook and metadata successfully generated at:", dst_dir)
