from numeric import get
from group import GroupFromNumeric

Num = get("tropical_min_plus")    # carica algebra min-plus
G = GroupFromNumeric(Num)

a, b = 2.0, 5.0
print("add:", G.add(a, b))
print("max:", G.max(a, b))
print("compare:", G.compare(a, b))

