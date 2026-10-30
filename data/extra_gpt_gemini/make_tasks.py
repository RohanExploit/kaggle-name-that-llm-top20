"""Build Kaggle Benchmarks generation tasks (one task file per competition label) for extra training passages.

Prompts come from the public prompt bank of seyitkaangunes/ntl-extra-api-passages (CC BY 4.0, 1,000 new prompts, the same
prompts as its 4,998 Claude passages), so every label answers the same prompts. Request = the published protocol of that
dataset: "<prompt> Write a single paragraph of about N words, with no title or headings.", no system prompt, one fresh chat
per call. N is drawn per (prompt, label) from that label's word counts in the competition train set (numbers only).
`setting` = "default" (provider defaults) or a kbench reasoning level ("none", "low", "medium", "high").
Each task checks the model it runs on and generates nothing on any other model (protects the free quota).

Usage: python3 src/gen/make_tasks.py <batch name> <labels comma-separated> <settings comma-separated> <prompt start> <prompt count>
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLUG = {  # competition label -> Kaggle Model Proxy model
    "gpt-5.5": "openai/gpt-5.5-2026-04-23", "gpt-5.6-luna": "openai/gpt-5.6-luna", "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra", "gpt-6-astra": "openai/gpt-6-astra",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview", "gemini-3.6-flash": "google/gemini-3.6-flash",
    "gemini-3.8-flash": "google/gemini-3.8-flash",
    "claude-opus-4-6": "anthropic/claude-opus-4-6@default", "claude-opus-4-8": "anthropic/claude-opus-4-8@default",
    "claude-opus-5": "anthropic/claude-opus-5@default", "claude-sonnet-5": "anthropic/claude-sonnet-5@default",
}

TEMPLATE = r'''# Name That LLM — extra training passages for label {label} ({batch}).
# Prompts: public prompt bank of seyitkaangunes/ntl-extra-api-passages (CC BY 4.0). Sequential raw calls through the
# task-supplied Kaggle proxy client (no keys stored), provider defaults unless a reasoning level is set.
import json
import time

import kaggle_benchmarks as kbench

LABEL = {label!r}
EXPECT = {expect!r}  # the proxy model this task is meant for
ROWS = json.loads({rows!r})


@kbench.task(name={task!r}, description="Extra training passages for the Name That LLM competition")
def {func}(llm) -> int:
    model = getattr(llm, "model", None) or getattr(llm, "name", "")
    if model.split("@")[0] != EXPECT.split("@")[0] and EXPECT.split("/")[-1].split("@")[0] not in str(model):
        print(json.dumps({{"skipped": True, "model": str(model)}}), flush=True)
        return 0
    client = llm.client.with_options(max_retries=2, timeout=240)
    done, errors, spent = 0, 0, 0.0
    out = open(f"gen_{{LABEL}}.jsonl", "a")
    for row in ROWS:
        request = {{"model": model, "max_tokens": 16000, "stream": False, "messages": [{{"role": "user", "content":
                   row["prompt"] + f" Write a single paragraph of about {{row['target_words']}} words, with no title or headings."}}]}}
        if row["setting"] != "default":
            request["reasoning_effort"] = row["setting"]
        t0 = time.time()
        try:
            r = client.chat.completions.create(**request)
        except Exception as e:  # never print request internals
            errors += 1
            msg = f"{{type(e).__name__}}: {{getattr(e, 'status_code', '')}}"
            print(json.dumps({{"prompt_id": row["prompt_id"], "error": msg}}), flush=True)
            if "429" in msg or "quota" in str(e).lower() or errors >= 15:
                print("stopping: quota / rate limit / too many errors", flush=True)
                break
            time.sleep(5)
            continue
        u = r.usage.model_dump() if getattr(r, "usage", None) else {{}}
        cost = u.get("cost") or {{}}
        cost_usd = (sum(v for v in cost.values() if isinstance(v, (int, float))) / 1e9) if isinstance(cost, dict) else 0.0
        spent += cost_usd
        rec = {{"label": LABEL, "model_seen": str(getattr(r, "model", model)), "prompt_id": row["prompt_id"], "setting": row["setting"],
               "target_words": int(row["target_words"]), "text": r.choices[0].message.content or "",
               "finish_reason": r.choices[0].finish_reason, "usage": u, "cost_usd": cost_usd, "latency_s": round(time.time() - t0, 2)}}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done += 1
    print(f"completed {{done}} / {{len(ROWS)}}, errors {{errors}}, cost ${{spent:.4f}}", flush=True)
    return done


{func}.run(kbench.llm)
'''


def main():
    batch, labels, settings, start, count = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4]), int(sys.argv[5])
    prompts = pd.read_csv(os.path.join(ROOT, "external", "data_extra", "api", "prompts.csv")).iloc[start:start + count]
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    wc = train.assign(w=train.text.str.split().str.len()).groupby("model").w.apply(np.array)
    out_dir = os.path.join(ROOT, "notebooks", "gen", batch)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for label in labels:
        rows = []
        for _, p in prompts.iterrows():
            seed = int(hashlib.sha256(f"{p.prompt_id}|{label}".encode()).hexdigest()[:8], 16)
            n = int(np.clip(np.random.default_rng(seed).choice(wc[label]), 36, 274))
            for s in settings:
                rows.append({"prompt_id": p.prompt_id, "prompt": p.prompt, "target_words": n, "setting": s})
        task = f"ntl-{batch}-{label.replace('.', '-')}"
        func = "gen_" + label.replace("-", "_").replace(".", "_")
        src = TEMPLATE.format(label=label, batch=batch, expect=SLUG[label], rows=json.dumps(rows), task=task, func=func)
        path = os.path.join(out_dir, f"{task}.py")
        open(path, "w").write(src)
        manifest.append({"task": task, "label": label, "model": SLUG[label], "file": path, "rows": len(rows)})
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
