import hashlib
data = b"row1\nrow2\nrow3"
print(hashlib.sha256(data).hexdigest())
