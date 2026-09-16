from itertools import islice
def batches(items, size):
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch
for batch in batches(range(10), 3):
    print(batch)
