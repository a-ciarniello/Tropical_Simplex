from __future__ import annotations
import numpy as np
from fractions import Fraction
from typing import Any, Dict, Type

class NumericBase:
    """Base class replicating the OCaml Numeric.T signature."""

    zero: Any

    def max(self, x, y): raise NotImplementedError
    def min(self, x, y): raise NotImplementedError
    def add(self, x, y): raise NotImplementedError
    def neg(self, x): raise NotImplementedError
    def mul(self, x, y): raise NotImplementedError
    def div(self, x, y): raise NotImplementedError
    def pow(self, x, n: int): raise NotImplementedError
    def compare(self, x, y) -> int: raise NotImplementedError

    def of_int(self, x: int): raise NotImplementedError
    def of_string(self, s: str): raise NotImplementedError
    def to_string(self, x): raise NotImplementedError


# ==========================================================
#  Cast of OCAML_INT in NumericInt
# ==========================================================
class NumericInt(NumericBase):
    zero = 0

    def max(self, x, y): return max(x, y)
    def min(self, x, y): return min(x, y)
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y): return x // y  # integer division
    def pow(self, x, n): return x ** n
    def compare(self, x, y): return (x > y) - (x < y)

    def of_int(self, x): return int(x)
    def of_string(self, s): return int(s)
    def to_string(self, x): return str(x)


# ==========================================================
# Cast of OCAML_FLOAT in NumericFloat (NumPy-based)
# ==========================================================
class NumericFloat(NumericBase):
    zero = 0.0

    def max(self, x, y): return np.maximum(x, y)
    def min(self, x, y): return np.minimum(x, y)
    def add(self, x, y): return np.add(x, y)
    def neg(self, x): return np.negative(x)
    def mul(self, x, y): return np.multiply(x, y)
    def div(self, x, y): return np.divide(x, y)
    def pow(self, x, n): return np.power(x, n)
    def compare(self, x, y):
        return int((x > y) - (x < y))  # scalar fallback if needed

    def of_int(self, x): return float(x)
    def of_string(self, s): return float(s)
    def to_string(self, x): return str(float(x))


# ==========================================================
#  Cast of OCAML_BIG_INT in NumericBigInt (based on Python int)
# ==========================================================
class NumericBigInt(NumericBase):
    zero = 0

    def max(self, x, y): return max(x, y)
    def min(self, x, y): return min(x, y)
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y): return x // y  # integer division
    def pow(self, x, n): return pow(x, n)
    def compare(self, x, y): return (x > y) - (x < y)

    def of_int(self, x): return int(x)
    def of_string(self, s): return int(s)
    def to_string(self, x): return str(x)


# ==========================================================
# Cast of OCAML_BIG_RAT in NumericBigRat (based on fractions.Fraction)
# ==========================================================
class NumericBigRat(NumericBase):
    zero = Fraction(0, 1)

    def max(self, x, y): return x if x >= y else y
    def min(self, x, y): return x if x <= y else y
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y):
        if y == 0:
            raise ZeroDivisionError("NumericBigRat: division by zero")
        return x / y
    def pow(self, x, n):
        if n >= 0:
            return x ** n
        else:
            return Fraction(1, 1) / (x ** -n)
    def compare(self, x, y): return (x > y) - (x < y)

    def of_int(self, x): return Fraction(x, 1)
    def of_string(self, s):
        if "/" in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den))
        else:
            return Fraction(int(s), 1)
    def to_string(self, x): return str(x)

# ==========================================================
# Tropical semiring: min-plus algebra
# ==========================================================
class TropicalNumericMinPlus(NumericBase):
    """Implements tropical (min,+) semiring:
    ⊕ = min, ⊗ = +, zero = +∞, one = 0
    """


    zero = np.inf  # tropical additive identity
    one = 0.0      # multiplicative identity

    def max(self, x, y):  # sometimes used for comparisons
        return np.maximum(x, y)

    def min(self, x, y):
        return np.minimum(x, y)

    def add(self, x, y):
        """Tropical addition: min(x, y)."""
        return np.minimum(x, y)

    def neg(self, x):
        """No proper additive inverse in tropical algebra.
        We'll define it as 'raise' for safety."""
        raise NotImplementedError("Tropical semiring has no additive inverse")

    def mul(self, x, y):
        """Tropical multiplication: x + y."""
        return np.add(x, y)

    def div(self, x, y):
        """Tropical division: x - y (since multiplication is +)."""
        return np.subtract(x, y)

    def pow(self, x, n):
        """Repeated tropical multiplication: n * x"""
        return np.add.reduce([x] * n)

    def compare(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.where(x > y, 1, np.where(x < y, -1, 0))
        else:
            return int((x > y) - (x < y))

    def of_int(self, x): return float(x)
    def of_string(self, s): return float(s)
    def to_string(self, x): return str(x)



# ==========================================================
#  Registry -- Get() interface
# ==========================================================
NUMERIC_MODULES: Dict[str, Type[NumericBase]] = {
    "ocaml_int": NumericInt,
    "ocaml_float": NumericFloat,
    "ocaml_big_int": NumericBigInt,
    "ocaml_big_rat": NumericBigRat,
    "tropical_min_plus": TropicalNumericMinPlus,
}


def get(name: str) -> NumericBase:
    try:
        return NUMERIC_MODULES[name]()  # return instance
    except KeyError:
        raise ValueError(f"{name}: unknown type of numerical data")


def get_name_of_modules():
    return list(NUMERIC_MODULES.keys())
