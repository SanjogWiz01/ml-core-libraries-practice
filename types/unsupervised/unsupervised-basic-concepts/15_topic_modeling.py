"""TOPIC MODELING — summarize word co-occurrence patterns across documents.

Latent Dirichlet Allocation (LDA) treats each document as a mixture of topics
and each topic as a word-probability distribution. It is a generative model,
not a classifier. A person interprets topic meaning; vocabulary, document length,
preprocessing, and topic count affect output. Inspect representative documents
as well as top words. This uses scikit-learn's variational Bayes LDA.

IMPLEMENTATION: CountVectorizer builds document-term counts; LDA fits topic-word
distributions; components_ and transform show top words and document mixtures.
"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = ["team scored goal match league coach player football",
        "goalkeeper saved goal in the match team won",
        "league coach planned player training for the next football match",
        "new phone camera battery screen device app technology",
        "software update improved phone camera and mobile app battery",
        "device makers build faster processor and screen technology",
        "recipe pasta tomato garlic kitchen cook dinner sauce",
        "chef prepared dinner with tomato pasta and fresh garlic",
        "kitchen recipe uses olive oil herbs and vegetables to cook"]
vectorizer = CountVectorizer(stop_words="english", min_df=1)
X = vectorizer.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=3, max_iter=30, learning_method="batch",
                                random_state=41, doc_topic_prior=.2, topic_word_prior=.1).fit(X)
words = np.array(vectorizer.get_feature_names_out())
for i, weights in enumerate(lda.components_):
    print(f"Topic {i}: {', '.join(words[np.argsort(weights)[-6:][::-1]])}")
print("Document topic proportions:\n", np.round(lda.transform(X), 2))
