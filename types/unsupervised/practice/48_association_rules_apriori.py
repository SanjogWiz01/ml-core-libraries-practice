"""48. Association rules: mine co-occurring items with a from-scratch Apriori."""
from itertools import combinations

transactions = [
    {"bread", "milk", "eggs"},
    {"bread", "diapers", "beer", "eggs"},
    {"milk", "diapers", "beer", "cola"},
    {"bread", "milk", "diapers", "beer"},
    {"bread", "milk", "diapers", "cola"},
]


def support(itemset):
    return sum(itemset <= t for t in transactions) / len(transactions)


def next_candidates(prev, size):
    out = set()
    for a, b in combinations(sorted(prev, key=sorted), 2):
        union = a | b
        if len(union) == size:
            out.add(union)
    return out


min_support = 0.4
frequent = {frozenset([i]) for t in transactions for i in t}
frequent = {s for s in frequent if support(s) >= min_support}

all_frequent = set(frequent)
size = 2
while frequent:
    frequent = {c for c in next_candidates(frequent, size) if support(c) >= min_support}
    all_frequent |= frequent
    size += 1

print(f"frequent itemsets (support >= {min_support}):")
for s in sorted(all_frequent, key=lambda x: (len(x), sorted(x))):
    print(f"  {sorted(s)}  support={support(s):.2f}")

print("\nassociation rules (confidence >= 0.7):")
for s in sorted(all_frequent, key=lambda x: (len(x), sorted(x))):
    if len(s) < 2:
        continue
    for item in s:
        antecedent = s - {item}
        confidence = support(s) / support(antecedent)
        lift = confidence / support({item})
        if confidence >= 0.7:
            print(f"  {sorted(antecedent)} -> {item}  conf={confidence:.2f} lift={lift:.2f}")
