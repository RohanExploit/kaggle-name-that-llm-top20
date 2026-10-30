import pandas as pd
import numpy as np

print("Loading master_train_pool.parquet...")
df = pd.read_parquet(r'E:\projects\kaggle\data\master_train_pool.parquet')
print(f"Total rows: {len(df)}")

df['text_str'] = df['text'].fillna('').astype(str)
df['curly'] = df['text_str'].apply(lambda t: t.count('“') + t.count('”') + t.count('‘') + t.count('’'))
df['em_dash'] = df['text_str'].apply(lambda t: t.count('—'))

print("\nCURLY QUOTES BY MODEL FAMILY:")
for fam in ['claude', 'glm', 'gpt', 'gemini', 'qwen', 'kimi']:
    sub = df[df['model'].str.startswith(fam)]
    has_curly = (sub['curly'] > 0).mean() * 100
    mean_curly = sub['curly'].mean()
    print(f"  {fam:<8}: {has_curly:5.2f}% of rows have curly quotes (mean: {mean_curly:.3f})")

print("\nEM-DASH BY MODEL FAMILY:")
for fam in ['claude', 'glm', 'gpt', 'gemini', 'qwen', 'kimi']:
    sub = df[df['model'].str.startswith(fam)]
    has_em = (sub['em_dash'] > 0).mean() * 100
    mean_em = sub['em_dash'].mean()
    print(f"  {fam:<8}: {has_em:5.2f}% of rows have em-dash (mean: {mean_em:.3f})")
