import random
random.seed(42)
a = [random.random() for _ in range(5)]
random.seed(42)
b = [random.random() for _ in range(5)]
print(a)
print(a == b)
