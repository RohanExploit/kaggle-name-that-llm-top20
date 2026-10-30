import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max"
]

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def preprocess(view, text):
    if view == "shape":
        return re.sub(r"[^\W\d_]+", lambda match: "X" if match[0].isupper() else "x", text)
    if view in ("small", "function"):
        return " ".join(word.lower() if word.lower() in FUNCTION_WORDS or not word.isalnum() else "CONTENT"
                        for word in re.findall(r"\w+|[^\w\s]", text))
    return text

def main():
    print("Loading rare1516_svc readout and vocabularies...")
    path = r"E:\projects\kaggle\scratch\rare_class_svc\rare1516_svc"
    with open(f"{path}/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    readout = np.load(f"{path}/readout.npz")
    coef = readout['coef']
    intercept = readout['intercept']
    target_labels = meta['labels']
    print(f"Readout coef: {coef.shape}, intercept: {intercept.shape}")
    print(f"Target labels ({len(target_labels)}): {target_labels[:5]}...")

    print("Loading train.csv...")
    df = pd.read_csv(r"E:\projects\kaggle\data\train.csv")
    texts = df['text'].fillna('').astype(str).tolist()
    y_true = np.array([target_labels.index(m) for m in df['model']])

    views = [("char", 1.0), ("word", 0.7), ("shape", 0.5), ("small", 0.5)]
    blocks = []

    for view, weight in views:
        print(f"Processing view: {view}...")
        with open(f"{path}/{view}-vocabulary.json", "r", encoding="utf-8") as f:
            voc = json.load(f)
        idf = np.load(f"{path}/{view}-idf.npy")
        
        vec = TfidfVectorizer(
            analyzer="char" if view == "shape" else "word" if view in ("small", "function") else view,
            vocabulary=voc,
            lowercase=False,
            sublinear_tf=True
        )
        vec.idf_ = idf
        
        pre = [preprocess(view, t) for t in texts]
        X = vec.transform(pre) * weight
        blocks.append(X)
        print(f"  {view} matrix: {X.shape}")

    X_all = hstack(blocks, format="csr")
    print(f"Combined feature matrix: {X_all.shape}")

    scores = X_all.dot(coef.T) + intercept
    y_pred = scores.argmax(axis=1)

    bacc = balanced_accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"Rare-Class SVC Balanced Accuracy on train.csv: {bacc:.4f}")
    print("="*50)

    print("\nPer-class accuracy:")
    for i, lab in enumerate(target_labels):
        mask = (y_true == i)
        acc = (y_pred[mask] == i).mean()
        print(f"  {lab:<25}: {acc:.4f} ({mask.sum()} samples)")

if __name__ == '__main__':
    main()
