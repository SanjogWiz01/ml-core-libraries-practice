from functools import partial
def scale(x, factor): return x * factor
double = partial(scale, factor=2)
print([double(x) for x in [1,2,3]])
