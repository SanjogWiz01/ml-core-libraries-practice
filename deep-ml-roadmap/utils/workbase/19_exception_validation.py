def parse_positive_int(value):
    try:
        n = int(value)
    except ValueError as e:
        raise ValueError("Value must be an integer") from e
    if n <= 0: raise ValueError("Value must be positive")
    return n
for x in ["10","0","abc"]:
    try: print(x, parse_positive_int(x))
    except ValueError as e: print("Error:", e)
