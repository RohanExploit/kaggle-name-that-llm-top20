import pandas as pd
import numpy as np

print("Loading master_train_pool.parquet for deep sibling analysis...")
df = pd.read_parquet(r'E:\projects\kaggle\data\master_train_pool.parquet')

# Sibling analysis within Gemini
print("\n" + "="*80)
print("DEEP SIBLING PROFILE: GEMINI FAMILY")
print("="*80)
gem = df[df['model'].str.startswith('gemini')].copy()
gem['len'] = gem['text'].fillna('').str.len()
gem['semicolon'] = gem['text'].fillna('').apply(lambda t: t.count(';'))
gem['curly'] = gem['text'].fillna('').apply(lambda t: t.count('“') + t.count('”') + t.count('‘') + t.count('’'))
gem['colon'] = gem['text'].fillna('').apply(lambda t: t.count(':'))
gem['hyphen'] = gem['text'].fillna('').apply(lambda t: t.count('-'))

gem_summary = gem.groupby('model')[['len', 'semicolon', 'curly', 'colon', 'hyphen']].mean()
print(gem_summary.round(3).to_string())

# Sibling analysis within Claude
print("\n" + "="*80)
print("DEEP SIBLING PROFILE: CLAUDE FAMILY")
print("="*80)
claude = df[df['model'].str.startswith('claude')].copy()
claude['len'] = claude['text'].fillna('').str.len()
claude['em_dash'] = claude['text'].fillna('').apply(lambda t: t.count('—'))
claude['semicolon'] = claude['text'].fillna('').apply(lambda t: t.count(';'))
claude['colon'] = claude['text'].fillna('').apply(lambda t: t.count(':'))
claude['hyphen'] = claude['text'].fillna('').apply(lambda t: t.count('-'))

claude_summary = claude.groupby('model')[['len', 'em_dash', 'semicolon', 'colon', 'hyphen']].mean()
print(claude_summary.round(3).to_string())

# Sibling analysis within GPT
print("\n" + "="*80)
print("DEEP SIBLING PROFILE: GPT FAMILY")
print("="*80)
gpt = df[df['model'].str.startswith('gpt')].copy()
gpt['len'] = gpt['text'].fillna('').str.len()
gpt['em_dash'] = gpt['text'].fillna('').apply(lambda t: t.count('—'))
gpt['curly'] = gpt['text'].fillna('').apply(lambda t: t.count('“') + t.count('”') + t.count('‘') + t.count('’'))
gpt['semicolon'] = gpt['text'].fillna('').apply(lambda t: t.count(';'))
gpt['colon'] = gpt['text'].fillna('').apply(lambda t: t.count(':'))

gpt_summary = gpt.groupby('model')[['len', 'em_dash', 'curly', 'semicolon', 'colon']].mean()
print(gpt_summary.round(3).to_string())
