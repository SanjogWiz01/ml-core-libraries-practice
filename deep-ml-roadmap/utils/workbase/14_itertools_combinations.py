from itertools import combinations, product
features = ["age", "income", "score"]
print(list(combinations(features, 2)))
print(list(product(["linear", "tree"], [0.1, 0.2])))
