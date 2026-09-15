import sqlite3
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE experiments(name TEXT, score REAL)")
cur.executemany("INSERT INTO experiments VALUES (?,?)",
                [("baseline",.81),("random_forest",.89),("logistic",.84)])
for row in cur.execute("SELECT name,score FROM experiments ORDER BY score DESC"):
    print(row)
conn.close()
