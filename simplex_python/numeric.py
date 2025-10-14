from __future__ import annotations
import numpy as np
from fractions import Fraction
from typing import Any, Dict, Type

class NumericBase:

    zero: Any
    one: Any

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
    one = 1

    def max(self, x, y): return max(x, y)
    def min(self, x, y): return min(x, y)
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y): return x // y
    def pow(self, x, n): return x ** n
    def compare(self, x, y): return (x > y) - (x < y)

    def of_int(self, x): return int(x)
    def of_string(self, s): return int(s)
    def to_string(self, x): return str(x)


# ==========================================================
# Cast of OCAML_FLOAT in NumericFloat
# ==========================================================
class NumericFloat(NumericBase):
    zero = 0.0
    one = 1.0

    def max(self, x, y): return np.maximum(x, y)
    def min(self, x, y): return np.minimum(x, y)
    def add(self, x, y): return np.add(x, y)
    def neg(self, x): return np.negative(x)
    def mul(self, x, y): return np.multiply(x, y)
    def div(self, x, y): return np.divide(x, y)
    def pow(self, x, n): return np.power(x, n)
    def compare(self, x, y):
        # Gestisce sia scalari che array numpy
        if isinstance(x, (list, np.ndarray)) or isinstance(y, (list, np.ndarray)):
            x_val = float(x) if not isinstance(x, (list, np.ndarray)) else float(x[0]) if len(x) > 0 else 0.0
            y_val = float(y) if not isinstance(y, (list, np.ndarray)) else float(y[0]) if len(y) > 0 else 0.0
        else:
            x_val, y_val = float(x), float(y)
        
        if x_val > y_val:
            return 1
        elif x_val < y_val:
            return -1
        else:
            return 0

    def of_int(self, x): return float(x)
    def of_string(self, s): return float(s)
    def to_string(self, x): return str(float(x))


# ==========================================================
#  Cast of OCAML_BIG_INT in NumericBigInt
# ==========================================================
class NumericBigInt(NumericBase):
    zero = 0
    one = 1

    def max(self, x, y): return max(x, y)
    def min(self, x, y): return min(x, y)
    def add(self, x, y): return x + y
    def neg(self, x): return -x
    def mul(self, x, y): return x * y
    def div(self, x, y): return x // y  
    def pow(self, x, n): return pow(x, n)
    def compare(self, x, y): return (x > y) - (x < y)

    def of_int(self, x): return int(x)
    def of_string(self, s): return int(s)
    def to_string(self, x): return str(x)


# ==========================================================
# Cast of OCAML_BIG_RAT in NumericBigRat
# ==========================================================
class NumericBigRat(NumericBase):
    zero = Fraction(0, 1)
    one = Fraction(1, 1)

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

    def max(self, x, y): 
        return np.maximum(x, y)

    def min(self, x, y):
        return np.minimum(x, y)

    def add(self, x, y):
        return np.minimum(x, y)

    def neg(self, x):
        raise NotImplementedError("Tropical semiring has no additive inverse")

    def mul(self, x, y):
        return np.add(x, y)

    def div(self, x, y):
        return np.subtract(x, y)

    def pow(self, x, n):
        return np.add.reduce([x] * n)

    def compare(self, x, y):
        """Compare two values/arrays element-wise.
        For arrays, uses lexicographic ordering (compares first differing element).
        Returns: 1 if x > y, -1 if x < y, 0 if x == y
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            # Convert both to arrays for consistent handling
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            
            # Handle same-shaped arrays with element-wise comparison
            if x_arr.shape == y_arr.shape:
                x_flat = x_arr.flatten()
                y_flat = y_arr.flatten()
                
                # Lexicographic comparison: find first differing element
                for i in range(len(x_flat)):
                    if x_flat[i] > y_flat[i]:
                        return 1
                    elif x_flat[i] < y_flat[i]:
                        return -1
                return 0  # All elements are equal
            else:
                # Different shapes: compare by broadcasting rules
                # This is a fallback for when shapes don't match
                try:
                    diff = x_arr - y_arr
                    diff_flat = diff.flatten()
                    for val in diff_flat:
                        if val > 0:
                            return 1
                        elif val < 0:
                            return -1
                    return 0
                except (ValueError, TypeError):
                    # If subtraction fails, compare sizes as last resort
                    if x_arr.size > y_arr.size:
                        return 1
                    elif x_arr.size < y_arr.size:
                        return -1
                    else:
                        return 0
        else:
            # Scalar comparison
            if x > y:
                return 1
            elif x < y:
                return -1
            else:
                return 0

    def of_int(self, x): return float(x)
    def of_string(self, s): return float(s)
    def to_string(self, x): return str(x)

# ==========================================================
# Tropical semiring: max-plus algebra
# ==========================================================

class TropicalNumericMaxPlus(NumericBase):
    """Implements tropical (max,+) semiring:
    ⊕ = max, ⊗ = +, zero = -∞, one = 0
    """

    zero = -np.inf  # tropical additive identity
    one = 0.0       # multiplicative identity

    def max(self, x, y):
        return np.maximum(x, y)

    def min(self, x, y):
        return np.minimum(x, y)

    def add(self, x, y):
        return np.maximum(x, y)

    def neg(self, x):
        raise NotImplementedError("Tropical semiring has no additive inverse")

    def mul(self, x, y):
        return np.add(x, y)

    def div(self, x, y):
        return np.subtract(x, y)

    def pow(self, x, n):
        return np.add.reduce([x] * n)

    def compare(self, x, y):
        """Compare two values/arrays element-wise.
        For arrays, uses lexicographic ordering (compares first differing element).
        Returns: 1 if x > y, -1 if x < y, 0 if x == y
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            # Convert both to arrays for consistent handling
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            
            # Handle same-shaped arrays with element-wise comparison
            if x_arr.shape == y_arr.shape:
                x_flat = x_arr.flatten()
                y_flat = y_arr.flatten()
                
                # Lexicographic comparison: find first differing element
                for i in range(len(x_flat)):
                    if x_flat[i] > y_flat[i]:
                        return 1
                    elif x_flat[i] < y_flat[i]:
                        return -1
                return 0  # All elements are equal
            else:
                # Different shapes: compare by broadcasting rules
                # This is a fallback for when shapes don't match
                try:
                    diff = x_arr - y_arr
                    diff_flat = diff.flatten()
                    for val in diff_flat:
                        if val > 0:
                            return 1
                        elif val < 0:
                            return -1
                    return 0
                except (ValueError, TypeError):
                    # If subtraction fails, compare sizes as last resort
                    if x_arr.size > y_arr.size:
                        return 1
                    elif x_arr.size < y_arr.size:
                        return -1
                    else:
                        return 0
        else:
            # Scalar comparison
            if x > y:
                return 1
            elif x < y:
                return -1
            else:
                return 0

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
    "tropical_max_plus": TropicalNumericMaxPlus,
}


def get(name: str) -> NumericBase:
    try:
        return NUMERIC_MODULES[name]()
    except KeyError:
        raise ValueError(f"{name}: unknown type of numerical data")


def get_name_of_modules():
    return list(NUMERIC_MODULES.keys())
