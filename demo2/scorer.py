import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class RobertaAnchorScorer:
    def __init__(self, lexicon_csv_path, threshold=0.7):
        # Load lexicon in format: word,hypo,hyper,flow
        df = pd.read_csv(lexicon_csv_path)
        self.lexicon = {}
        for _, row in df.iterrows():
            word = row["word"].lower().strip()
            self.lexicon[word] = {
                "hypo": float(row.get("hypo", 0.0)),
                "hyper": float(row.get("hyper", 0.0)),
                "flow": float(row.get("flow", 0.0))
            }

        # Consider all words in lexicon
        self.anchor_words = list(self.lexicon.keys())

        # Load SentenceTransformer (better semantic embeddings than raw RoBERTa)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

        # Cache for embeddings
        self.cache = {}

        # Precompute embeddings for all anchor words
        self.anchor_embs = self._encode_texts(self.anchor_words)

    def _encode_texts(self, texts):
        new_texts = [t for t in texts if t not in self.cache]
        if new_texts:
            embs = self.model.encode(new_texts, convert_to_numpy=True)
            for t, e in zip(new_texts, embs):
                self.cache[t] = e
        return np.array([self.cache[t] for t in texts])

    def score_sentence(self, sentence):
        STOPWORDS = set(stopwords.words("english"))
        tokens = [t for t in word_tokenize(sentence.lower()) if t.isalpha() and t not in STOPWORDS]

        if not tokens:
            return {"percentages": {"hypo": 0, "hyper": 0, "flow": 0},
                    "dominant": None,
                    "label": "No valid tokens",
                    "matched": []}

        token_embs = self._encode_texts(tokens)

        scores = {"hypo": 0, "hyper": 0, "flow": 0}
        matched = []

        # Compare each token with ALL anchors
        for i, tok in enumerate(tokens):
            sims = cosine_similarity([token_embs[i]], self.anchor_embs)[0]
            for idx, sim in enumerate(sims):
                if sim >= self.threshold:
                    anchor = self.anchor_words[idx]
                    contrib = self.lexicon[anchor]
                    for state, val in contrib.items():
                        scores[state] += val * sim  # weight by similarity
                    matched.append({
                        "token": tok,
                        "matched_anchor": anchor,
                        "similarity": float(sim)
                    })

        # Normalize into percentages
        total = sum(scores.values())
        if total == 0:
            percentages = {s: 0 for s in scores}
            dominant = None
            label = "No relevant keywords found"
        else:
            percentages = {s: round(v / total * 100, 2) for s, v in scores.items()}
            dominant = max(percentages, key=percentages.get)
            label = (f"Mixed: {percentages['hypo']}% hypo, "
                     f"{percentages['hyper']}% hyper, "
                     f"{percentages['flow']}% flow "
                     f"(dominant: {dominant})")

        return {
            "percentages": percentages,
            "dominant": dominant,
            "label": label,
            "matched": matched
        }
