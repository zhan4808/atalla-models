import logging

import demo1

from ppci.lang.python import load_py

logging.basicConfig(level=logging.DEBUG)

with open("demo1.py") as f:
    m2 = load_py(f)

for x in range(20):
    print(x, m2.a(x, 2), demo1.a(x, 2))
