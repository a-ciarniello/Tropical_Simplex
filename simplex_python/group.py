from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, TypeVar, Generic
import numeric


# ==========================================================
# Base class: additive group
# ==========================================================
class AdditiveGroup:
    """Additive (commutative) group."""

    def zero(self) -> Any:
        raise NotImplementedError

    def add(self, x, y):
        raise NotImplementedError

    def substract(self, x, y):
        return self.add(x, self.neg(y))

    def neg(self, x):
        raise NotImplementedError

    def mul(self, x, y):
        raise NotImplementedError

    def sum(self, lst: List[Any]):
        if not lst:
            raise ValueError("sum: empty list")
        result = lst[0]
        for elem in lst[1:]:
            result = self.add(result, elem)
        return result

    def to_string(self, x) -> str:
        return str(x)

    operation_string = "+"


# ==========================================================
# Ordered group (Ordered in OCaml)
# ==========================================================
class OrderedGroup(AdditiveGroup):
    """Additive group with ordering and max."""

    def compare(self, x, y) -> int:
        raise NotImplementedError

    def max(self, x, y):
        c = self.compare(x, y)
        if c >= 0:
            return x
        return y


# ==========================================================
# Make(Numeric): build OrderedGroup from Numeric module
# ==========================================================
class GroupFromNumeric(OrderedGroup):
    """Equivalent to Group.Make(Numeric)."""

    def __init__(self, NumericModule: numeric.NumericBase):
        self.Numeric = NumericModule

    def zero(self):
        return self.Numeric.zero

    def add(self, x, y):
        return self.Numeric.add(x, y)

    def neg(self, x):
        return self.Numeric.neg(x)

    def mul(self, x, y):
        return self.Numeric.mul(x, y)

    def compare(self, x, y):
        return self.Numeric.compare(x, y)

    def to_string(self, x):
        return self.Numeric.to_string(x)

    def max(self, x, y):
        return self.Numeric.max(x, y)


# ==========================================================
# Int group
# ==========================================================
class IntGroup(OrderedGroup):
    def zero(self):
        return 0

    def add(self, x, y):
        return x + y

    def neg(self, x):
        return -x

    def mul(self, x, y):
        return x * y

    def compare(self, x, y):
        return (x > y) - (x < y)

    def max(self, x, y):
        return max(x, y)

    def to_string(self, x):
        return str(x)


# ==========================================================
# ReverseOrder(Group)
# ==========================================================
class ReverseOrder(OrderedGroup):
    """Inverts the order of another OrderedGroup."""

    def __init__(self, Group: OrderedGroup):
        self.G = Group

    def zero(self):
        return self.G.zero()

    def add(self, x, y):
        return self.G.add(x, y)

    def neg(self, x):
        return self.G.neg(x)

    def compare(self, x, y):
        c = self.G.compare(x, y)
        return -c if c != 0 else 0

    def max(self, x, y):
        c = self.compare(x, y)
        return x if c >= 0 else y

    def to_string(self, x):
        return self.G.to_string(x)


# ==========================================================
# Cartesian Product (pair)
# ==========================================================
@dataclass
class CartesianProduct(OrderedGroup):
    def __init__(self, G: OrderedGroup, H: OrderedGroup):
        self.G = G
        self.H = H

    def zero(self):
        return (self.G.zero(), self.H.zero())

    def add(self, a, b):
        g1, h1 = a
        g2, h2 = b
        return (self.G.add(g1, g2), self.H.add(h1, h2))

    def neg(self, a):
        g, h = a
        return (self.G.neg(g), self.H.neg(h))

    def mul(self, a, b):
        g1, h1 = a
        g2, h2 = b
        return (self.G.mul(g1, g2), self.H.mul(h1, h2))

    def compare(self, a, b):
        g1, h1 = a
        g2, h2 = b
        c = self.G.compare(g1, g2)
        if c == 0:
            return self.H.compare(h1, h2)
        return c

    def max(self, a, b):
        return a if self.compare(a, b) >= 0 else b

    def from_entries(self, g, h):
        return (g, h)

    def first(self, a):
        return a[0]

    def second(self, a):
        return a[1]

    def to_string(self, a):
        g, h = a
        return f"[|{self.G.to_string(g)}; {self.H.to_string(h)}|]"


# ==========================================================
# Cartesian Power (sparse)
# ==========================================================
@dataclass
class CartesianPowerSparse(OrderedGroup):
    """List of (index, value) pairs sorted by index."""
    G: OrderedGroup

    def zero(self):
        return []

    def add(self, x: List[Tuple[int, Any]], y: List[Tuple[int, Any]]):
        """Merge two sorted sparse vectors."""
        res = []
        i = j = 0
        while i < len(x) and j < len(y):
            ix, vx = x[i]
            iy, vy = y[j]
            if ix == iy:
                s = self.G.add(vx, vy)
                if self.G.compare(s, self.G.zero()) != 0:
                    res.append((ix, s))
                i += 1
                j += 1
            elif ix < iy:
                res.append((ix, vx))
                i += 1
            else:
                res.append((iy, vy))
                j += 1
        res.extend(x[i:])
        res.extend(y[j:])
        return res

    def neg(self, x):
        return [(i, self.G.neg(v)) for i, v in x]

    def mul(self, x: List[Tuple[int, Any]], y: List[Tuple[int, Any]]):
        """Element-wise multiplication of two sparse vectors."""
        res = []
        i = j = 0
        while i < len(x) and j < len(y):
            ix, vx = x[i]
            iy, vy = y[j]
            if ix == iy:
                s = self.G.mul(vx, vy)
                if self.G.compare(s, self.G.zero()) != 0:
                    res.append((ix, s))
                i += 1
                j += 1
            elif ix < iy:
                i += 1
            else:
                j += 1
        return res

    def substract(self, x, y):
        return self.add(x, self.neg(y))

    def get(self, x, index: int):
        for i, v in x:
            if i == index:
                return v
        return self.G.zero()

    def from_list(self, lst: List[Tuple[int, Any]]):
        """Sort and filter out zero entries."""
        lst_sorted = sorted(lst, key=lambda t: t[0])
        cleaned = []
        prev = -1
        for i, v in lst_sorted:
            if i == prev:
                raise ValueError(f"Duplicate index {i} in sparse vector.")
            prev = i
            if self.G.compare(v, self.G.zero()) != 0:
                cleaned.append((i, v))
        return cleaned

    def compare(self, x, y):
        """Lexicographic compare of sparse lists."""
        i = j = 0
        while i < len(x) or j < len(y):
            if i == len(x):
                iy, vy = y[j]
                return -self.G.compare(vy, self.G.zero())
            if j == len(y):
                ix, vx = x[i]
                return self.G.compare(vx, self.G.zero())

            ix, vx = x[i]
            iy, vy = y[j]
            if ix < iy:
                c = self.G.compare(vx, self.G.zero())
                return c if c != 0 else self.compare(x[i+1:], y[j:])
            elif iy < ix:
                c = self.G.compare(vy, self.G.zero())
                return -c if c != 0 else self.compare(x[i:], y[j+1:])
            else:
                c = self.G.compare(vx, vy)
                if c != 0:
                    return c
                i += 1
                j += 1
        return 0

    def max(self, x, y):
        return x if self.compare(x, y) >= 0 else y

    def to_string(self, x):
        return "[" + "; ".join(f"({i}, {self.G.to_string(v)})" for i, v in x) + "]"
