from functools import lru_cache
@lru_cache(maxsize=128)
def expensive_feature(x):
    print("computing", x)
    return x*x
print(expensive_feature(10))
print(expensive_feature(10))
print(expensive_feature.cache_info())
