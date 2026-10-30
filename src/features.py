"""
Stylometric, typographic, punctuation, and rhythm feature extraction for Name That LLM.
Engineered specifically to separate frontier LLM families and sibling models.
"""

import re
import string
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_stylometric_features(texts: List[str]) -> pd.DataFrame:
    """
    Extracts dense linguistic, typographic, and stylistic features for each passage.
    """
    rows = []

    for text in texts:
        t = str(text)
        n_chars = len(t)
        words = re.findall(r"\b\w+\b", t)
        n_words = max(len(words), 1)

        # Sentence extraction
        sentences = re.split(r"[.!?]+(?:\s+|$)", t)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_sentences = max(len(sentences), 1)
        sentence_word_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences] or [n_words]

        # Punctuation counts
        count_emdash = t.count("—") + t.count("–") + t.count("--")
        count_curly_apostrophe = t.count("’") + t.count("‘")
        count_straight_apostrophe = t.count("'")
        count_curly_dquote = t.count("“") + t.count("”")
        count_straight_dquote = t.count('"')
        count_semicolon = t.count(";")
        count_colon = t.count(":")
        count_comma = t.count(",")
        count_period = t.count(".")
        count_question = t.count("?")
        count_exclamation = t.count("!")
        count_hyphen = t.count("-")
        count_paren = t.count("(") + t.count(")")
        count_ellipsis = t.count("...") + t.count("…")
        count_asterisk = t.count("*")  # markdown italics or bold markers

        # Total apostrophes & quotes ratios
        total_apos = count_curly_apostrophe + count_straight_apostrophe
        total_dquotes = count_curly_dquote + count_straight_dquote

        curly_apos_ratio = count_curly_apostrophe / total_apos if total_apos > 0 else 0.0
        curly_dquote_ratio = count_curly_dquote / total_dquotes if total_dquotes > 0 else 0.0

        # Word lengths
        word_lengths = [len(w) for w in words] or [1]
        unique_words = set(w.lower() for w in words)
        type_token_ratio = len(unique_words) / n_words

        # Capitalization
        cap_words = [w for w in words if w.istitle()]
        all_caps_words = [w for w in words if len(w) > 1 and w.isupper()]

        # Transition / discourse markers commonly preferred by specific models
        lower_t = t.lower()
        has_furthermore = 1.0 if "furthermore" in lower_t else 0.0
        has_moreover = 1.0 if "moreover" in lower_t else 0.0
        has_delve = 1.0 if "delve" in lower_t else 0.0
        has_tapestry = 1.0 if "tapestry" in lower_t else 0.0
        has_testament = 1.0 if "testament" in lower_t else 0.0
        has_crucial = 1.0 if "crucial" in lower_t else 0.0
        has_vital = 1.0 if "vital" in lower_t else 0.0
        has_ultimately = 1.0 if "ultimately" in lower_t else 0.0
        has_indeed = 1.0 if "indeed" in lower_t else 0.0

        # Per 100 words normalized rates
        rate_emdash = (count_emdash / n_words) * 100
        rate_curly_apos = (count_curly_apostrophe / n_words) * 100
        rate_straight_apos = (count_straight_apostrophe / n_words) * 100
        rate_semicolon = (count_semicolon / n_words) * 100
        rate_colon = (count_colon / n_words) * 100
        rate_comma = (count_comma / n_words) * 100
        rate_paren = (count_paren / n_words) * 100
        rate_hyphen = (count_hyphen / n_words) * 100
        rate_asterisk = (count_asterisk / n_words) * 100

        feat = {
            # Lengths & counts
            "n_words": n_words,
            "n_chars": n_chars,
            "n_sentences": n_sentences,
            "char_per_word": n_chars / n_words,
            # Sentence rhythm stats
            "mean_sentence_len": float(np.mean(sentence_word_lengths)),
            "std_sentence_len": float(np.std(sentence_word_lengths)),
            "min_sentence_len": float(np.min(sentence_word_lengths)),
            "max_sentence_len": float(np.max(sentence_word_lengths)),
            # Word length stats & vocabulary richness
            "mean_word_len": float(np.mean(word_lengths)),
            "std_word_len": float(np.std(word_lengths)),
            "type_token_ratio": type_token_ratio,
            "capitalized_word_rate": len(cap_words) / n_words,
            "all_caps_word_rate": len(all_caps_words) / n_words,
            # Punctuation presence binary flags
            "has_emdash": 1.0 if count_emdash > 0 else 0.0,
            "has_curly_apostrophe": 1.0 if count_curly_apostrophe > 0 else 0.0,
            "has_straight_apostrophe": 1.0 if count_straight_apostrophe > 0 else 0.0,
            "has_curly_dquote": 1.0 if count_curly_dquote > 0 else 0.0,
            "has_semicolon": 1.0 if count_semicolon > 0 else 0.0,
            "has_colon": 1.0 if count_colon > 0 else 0.0,
            "has_ellipsis": 1.0 if count_ellipsis > 0 else 0.0,
            "has_asterisk": 1.0 if count_asterisk > 0 else 0.0,
            # Normalized rates per 100 words
            "rate_emdash": rate_emdash,
            "rate_curly_apos": rate_curly_apos,
            "rate_straight_apos": rate_straight_apos,
            "rate_semicolon": rate_semicolon,
            "rate_colon": rate_colon,
            "rate_comma": rate_comma,
            "rate_paren": rate_paren,
            "rate_hyphen": rate_hyphen,
            "rate_asterisk": rate_asterisk,
            # Punctuation style ratios
            "curly_apos_ratio": curly_apos_ratio,
            "curly_dquote_ratio": curly_dquote_ratio,
            # Model idiosyncrasies
            "has_furthermore": has_furthermore,
            "has_moreover": has_moreover,
            "has_delve": has_delve,
            "has_tapestry": has_tapestry,
            "has_testament": has_testament,
            "has_crucial": has_crucial,
            "has_vital": has_vital,
            "has_ultimately": has_ultimately,
            "has_indeed": has_indeed,
        }
        rows.append(feat)

    return pd.DataFrame(rows)


class FeaturePipeline:
    """
    Combined feature pipeline:
    1. Word-level TF-IDF (n-grams 1-2)
    2. Character-level TF-IDF char_wb (n-grams 2-5)
    3. Handcrafted stylometric / punctuation features
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 15000,
        char_ngram_range: Tuple[int, int] = (2, 5),
        char_max_features: int = 25000,
    ):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=True,
            lowercase=False,  # Keep case sensitive for capitalization markers
        )
        self.char_vec = TfidfVectorizer(
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            analyzer="char_wb",
            sublinear_tf=True,
            lowercase=False,
        )
        self.stylometric_feature_names: List[str] = []

    def fit_transform(self, texts: List[str]) -> Tuple[csr_matrix, pd.DataFrame]:
        """Fits vectorizers on training texts and returns combined sparse matrix and dense DataFrame."""
        x_word = self.word_vec.fit_transform(texts)
        x_char = self.char_vec.fit_transform(texts)
        df_dense = extract_stylometric_features(texts)
        self.stylometric_feature_names = df_dense.columns.tolist()

        x_dense = csr_matrix(df_dense.values.astype(np.float32))
        x_all = hstack([x_dense, x_word, x_char], format="csr")
        return x_all, df_dense

    def transform(self, texts: List[str]) -> Tuple[csr_matrix, pd.DataFrame]:
        """Transforms new texts using fitted vectorizers."""
        x_word = self.word_vec.transform(texts)
        x_char = self.char_vec.transform(texts)
        df_dense = extract_stylometric_features(texts)

        x_dense = csr_matrix(df_dense.values.astype(np.float32))
        x_all = hstack([x_dense, x_word, x_char], format="csr")
        return x_all, df_dense
