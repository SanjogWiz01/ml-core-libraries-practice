import os
print("Working directory:", os.getcwd())
print("Python env:", os.getenv("PYTHON", "not set"))
print("User:", os.getenv("USERNAME") or os.getenv("USER", "unknown"))
