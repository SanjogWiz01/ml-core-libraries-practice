from collections import defaultdict
rows = [("Ram","ML",85),("Sita","ML",92),("Hari","SQL",88)]
grouped = defaultdict(list)
for name, subject, score in rows:
    grouped[subject].append((name, score))
print(dict(grouped))
