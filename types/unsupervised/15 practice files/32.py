"""32. TruncatedSVD (LSA): latent semantic analysis of a document-term matrix."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the kitten chased the mouse",
    "the puppy chased its tail",
    "stocks and bonds and interest rates",
    "shares fell as bond yields rose",
]

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)

svd = TruncatedSVD(n_components=2, random_state=42)
X_lsa = svd.fit_transform(X)

terms = np.array(vec.get_feature_names_out())
for i, comp in enumerate(svd.components_):
    top = terms[np.abs(comp).argsort()[::-1][:4]]
    print(f"latent concept {i}: {', '.join(top)}")

print("explained variance ratio:", svd.explained_variance_ratio_.round(3))
print("document vectors (first 3):")
print(X_lsa[:3].round(3))
