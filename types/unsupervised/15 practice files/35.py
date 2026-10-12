"""35. Latent Dirichlet Allocation: probabilistic topic modelling."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "the team scored a goal and won the match",
    "the striker scored twice in the football final",
    "the coach praised the goalkeeper after the win",
    "the stock market rallied on strong earnings",
    "investors bought shares and bond yields fell",
    "the bank held interest rates steady today",
    "the midfielder scored in the second half",
    "equities and bonds moved on the inflation report",
]

vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2, random_state=42, max_iter=50)
doc_topic = lda.fit_transform(X)

terms = np.array(vec.get_feature_names_out())
for topic, comp in enumerate(lda.components_):
    top = terms[comp.argsort()[::-1][:5]]
    print(f"topic {topic}: {', '.join(top)}")

print("dominant topic per document:", doc_topic.argmax(axis=1))
