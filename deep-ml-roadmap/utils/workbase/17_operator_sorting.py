from operator import itemgetter
rows = [{"name":"Ram","score":82},{"name":"Sita","score":95},{"name":"Hari","score":88}]
print(sorted(rows, key=itemgetter("score"), reverse=True))
