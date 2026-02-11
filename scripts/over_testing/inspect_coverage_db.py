import sqlite3, os, sys
path = '.coverage'
if not os.path.exists(path):
    print('No .coverage file present')
    sys.exit(0)
con = sqlite3.connect(path)
cur = con.cursor()
cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
rows = cur.fetchall()
for name, sql in rows:
    print('TABLE:', name)
    print(sql)
con.close()
