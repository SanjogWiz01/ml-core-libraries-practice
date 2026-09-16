from collections import Counter
labels = ["spam", "ham", "spam", "promo", "spam", "ham"]
c = Counter(labels)
print(c)
print(c.most_common())
