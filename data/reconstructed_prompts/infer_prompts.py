"""Infer the generating prompt for each train prompt_id with Claude Sonnet 5 (Bedrock), then meta-analyse.
Outputs: data/prompts_inferred.csv, research/prompts_meta.json"""
import json, re, time
from concurrent.futures import ThreadPoolExecutor
import boto3, pandas as pd
from botocore.config import Config

MODEL = "us.anthropic.claude-sonnet-5"
br = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=300))
df = pd.read_csv("data/train.csv")

INSTR = """Below are {n} passages written by different AI models in response to the SAME hidden prompt (shuffled, model names withheld).
Reconstruct that prompt. Reply with ONLY a JSON object with keys:
"inferred_prompt": the most likely exact instruction given to the models (one or two sentences),
"topic": short topic name (e.g. "Jevons paradox"),
"domain": one of ["philosophy","physics","mathematics","biology","computer science","creative prose","history","economics","other"],
"task_type": one of ["explain concept","argue/opinion","describe/narrate","story/fiction","compare","define","how-to","other"],
"audience_or_style": any audience/register/style constraint you can infer (or "none evident"),
"length_instruction": your guess about any length instruction given the word counts (e.g. "target N words varies per passage", "single paragraph"),
"shared_features": common structural features across passages (openings, single paragraph, no headings, etc.),
"confidence": number 0-1.

{passages}"""

def call(text, max_tokens=1500):
    for attempt in range(6):
        try:
            r = br.converse(modelId=MODEL, messages=[{"role": "user", "content": [{"text": text}]}],
                            inferenceConfig={"maxTokens": max_tokens})
            return r["output"]["message"]["content"][0]["text"]
        except Exception as e:
            print("retry", attempt, str(e)[:120], flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("bedrock failed")

def parse(s):
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0))

def infer(pid):
    g = df[df.prompt_id == pid].sample(frac=1, random_state=0)
    passages = "\n\n".join(f"### Passage {i+1} ({len(t.split())} words)\n{t}" for i, t in enumerate(g.text))
    out = parse(call(INSTR.format(n=len(g), passages=passages)))
    wc = g.text.str.split().str.len()
    out.update(prompt_id=pid, n_passages=len(g), words_min=int(wc.min()), words_median=int(wc.median()), words_max=int(wc.max()))
    return out

pids = df.prompt_id.unique().tolist()
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(infer, pids))
res = pd.DataFrame(rows)
cols = ["prompt_id", "inferred_prompt", "topic", "domain", "task_type", "audience_or_style", "length_instruction",
        "shared_features", "confidence", "n_passages", "words_min", "words_median", "words_max"]
res[cols].to_csv("data/prompts_inferred.csv", index=False)
print(res.domain.value_counts(), res.task_type.value_counts(), sep="\n")

# ---- meta-analysis over all 150 reconstructed prompts ----
listing = "\n".join(f"- [{r.domain} | {r.task_type}] {r.inferred_prompt}  (words {r.words_min}-{r.words_max})" for r in res.itertuples())
META = f"""These are 150 reconstructed prompts that a competition host used to make 18 AI models each write one self-contained passage (40-250 words, length varies randomly).
Infer the host's generation protocol. Reply with ONLY a JSON object with keys:
"likely_template": the probable prompt template with placeholders (e.g. topic, word count),
"topic_generation": how topics seem to have been chosen (lists, categories, granularity),
"domain_mix": estimated proportions per domain,
"length_mechanism": how lengths were probably controlled,
"test_prompt_predictions": what unseen test prompts are likely to look like (domains, style),
"synthetic_data_recipe": a concrete recipe to generate more training data in the same style (template + topic list strategy),
"example_new_prompts": 20 new prompts in the same style that are NOT in the list.

{listing}"""
meta = parse(call(META, max_tokens=6000))
json.dump(meta, open("research/prompts_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1)[:3000])
