"""ASSOCIATION RULES — find co-occurrence patterns in transaction baskets.

Support is the fraction of baskets containing an itemset. Confidence P(B|A) is
the fraction of A-baskets also containing B. Lift divides confidence by overall
prevalence of B: above 1 indicates positive association relative to independence,
not causation. Apriori uses anti-monotonicity: a superset cannot be more frequent
than its subset. This brute-force teaching version is for tiny baskets; use a
production FP-Growth/Apriori implementation for scale.

IMPLEMENTATION: enumerate itemsets and compute supports; generate directional
rules and print support, confidence, and lift.
"""
from itertools import combinations

baskets = [set(row.split()) for row in ["bread milk", "bread diaper beer eggs",
    "milk diaper beer cola", "bread milk diaper beer", "bread milk diaper cola",
    "milk eggs", "bread beer", "bread milk beer", "diaper beer cola", "bread diaper milk"]]
min_support = .2
items = sorted(set.union(*baskets))
def support(itemset):
    return sum(itemset <= basket for basket in baskets) / len(baskets)
frequent = {}
for size in range(1, 4):
    for combo in combinations(items, size):
        itemset = frozenset(combo)
        value = support(itemset)
        if value >= min_support:
            frequent[itemset] = value
rules = []
for union, sup in frequent.items():
    for size in range(1, len(union)):
        for left_tuple in combinations(union, size):
            left = frozenset(left_tuple)
            right = union - left
            confidence = sup / support(left)
            lift = confidence / support(right)
            rules.append((lift, confidence, sup, left, right))
for lift, confidence, sup, left, right in sorted(rules, reverse=True)[:8]:
    print(f"{sorted(left)} -> {sorted(right)} | support={sup:.2f}, "
          f"confidence={confidence:.2f}, lift={lift:.2f}")
