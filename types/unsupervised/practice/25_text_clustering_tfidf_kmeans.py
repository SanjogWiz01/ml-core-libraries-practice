"""25. Text clustering: group unlabelled documents with TF-IDF + K-Means."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

docs = [
    "the team scored a goal in the final minute",
    "the striker scored twice in the match",
    "the goalkeeper saved a penalty",
    "the stock market rallied on strong earnings",
    "investors bought shares after the report",
    "interest rates were held steady by the bank",
    "the coach praised the defence after the win",
    "bond yields fell as inflation cooled",
]

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)

km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels = km.fit_predict(X)

terms = np.array(vec.get_feature_names_out())
for c in range(2):
    top = terms[km.cluster_centers_[c].argsort()[::-1][:4]]
    members = [docs[i] for i in np.where(labels == c)[0]]
    print(f"cluster {c} top terms: {', '.join(top)}")
    for m in members:
        print("   -", m)
