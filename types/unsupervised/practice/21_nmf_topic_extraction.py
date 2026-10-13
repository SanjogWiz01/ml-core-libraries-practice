"""21. NMF: non-negative matrix factorisation for topic extraction."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

docs = [
    "the cat sat on the mat",
    "the dog played in the park",
    "stock markets fell today",
    "interest rates rose sharply",
    "the cat chased the dog",
    "investors sold their shares",
    "a puppy and a kitten played",
    "the market rallied on earnings",
]

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)

nmf = NMF(n_components=2, init="nndsvd", random_state=42)
W = nmf.fit_transform(X)

terms = np.array(vec.get_feature_names_out())
for topic, comp in enumerate(nmf.components_):
    top = terms[comp.argsort()[::-1][:5]]
    print(f"topic {topic}: {', '.join(top)}")

print("document -> topic weights:")
print(W.round(2))
