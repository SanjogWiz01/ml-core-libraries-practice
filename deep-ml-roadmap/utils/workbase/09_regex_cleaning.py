import re
text = "Contact data.scientist@example.com or ml@example.org"
print(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
print(re.sub(r"[^A-Za-z0-9 ]+", " ", "ML!! Data--Science"))
