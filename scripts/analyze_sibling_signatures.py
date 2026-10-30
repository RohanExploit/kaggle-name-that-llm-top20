import pandas as pd
import numpy as np
import re
from collections import Counter

train_path = r'E:\projects\kaggle\data\train.csv'
df = pd.read_csv(train_path)

print(f"Dataset total samples: {len(df)}")
models = df['model'].unique()

def extract_features(text):
    features = {}
    features['char_len'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_len'] = features['char_len'] / max(features['word_count'], 1)
    
    # Stylistic / Punctuation Markers
    features['em_dash'] = text.count('—')
    features['en_dash'] = text.count('–')
    features['hyphen'] = text.count('-')
    features['ellipsis'] = text.count('…') + text.count('...')
    features['semicolon'] = text.count(';')
    features['colon'] = text.count(':')
    features['quotes_straight'] = text.count('"') + text.count("'")
    features['quotes_curly'] = text.count('“') + text.count('”') + text.count('‘') + text.count('’')
    
    # Markdown & Formatting
    features['markdown_headers'] = len(re.findall(r'^#+\s', text, re.MULTILINE))
    features['bullet_asterisk'] = len(re.findall(r'^\s*\*\s', text, re.MULTILINE))
    features['bullet_dash'] = len(re.findall(r'^\s*-\s', text, re.MULTILINE))
    features['numbered_list'] = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
    features['bold_text'] = len(re.findall(r'\*\*.*?\*\*', text))
    features['code_blocks'] = text.count('```')
    
    # Vocabulary & Case
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(features['char_len'], 1)
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(features['char_len'], 1)
    
    return features

data_feats = [extract_features(t) for t in df['text']]
feat_df = pd.DataFrame(data_feats)
feat_df['model'] = df['model']

# Group by family
def get_family(m):
    if 'claude' in m: return 'claude'
    if 'gpt' in m: return 'gpt'
    if 'gemini' in m: return 'gemini'
    if 'glm' in m: return 'glm'
    if 'qwen' in m: return 'qwen'
    return 'kimi'

feat_df['family'] = feat_df['model'].apply(get_family)

# Analyze Family-level differences
print("\n" + "="*80)
print("FAMILY-LEVEL STYLISTIC INVARIANTS (Means):")
print("="*80)
family_means = feat_df.groupby('family')[['char_len', 'quotes_curly', 'em_dash', 'semicolon', 'bullet_dash', 'bullet_asterisk', 'markdown_headers', 'bold_text']].mean()
print(family_means.round(3).to_string())

# Analyze Within-Family Sibling Differences
for fam in ['claude', 'gpt', 'gemini', 'glm']:
    print("\n" + "="*80)
    print(f"SIBLING ATTRIBUTION: FAMILY {fam.upper()} (Means):")
    print("="*80)
    sub = feat_df[feat_df['family'] == fam]
    sub_means = sub.groupby('model')[['char_len', 'word_count', 'quotes_curly', 'em_dash', 'semicolon', 'bullet_dash', 'bullet_asterisk', 'markdown_headers', 'bold_text']].mean()
    print(sub_means.round(3).to_string())
