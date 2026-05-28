"""Numeric backends that implement the ordered-group contract."""

from __future__ import annotations
import numpy as np
from fractions import Fraction
from decimal import Decimal
from typing import Any, Dict, Type



FLOAT_ATOL = 1e-9
FLOAT_RTOL = 1e-9


def _cmp_scalar_with_tol(x_val: float, y_val: float) -> int:
    if not np.isfinite(x_val) or not np.isfinite(y_val):
        return (x_val > y_val) - (x_val < y_val)
    diff = x_val - y_val
    tol = max(FLOAT_ATOL, FLOAT_RTOL * max(abs(x_val), abs(y_val)))
    if abs(diff) <= tol:
        return 0
    return 1 if diff > 0 else -1


def _cmp_ndarray_with_tol(x_arr: np.ndarray, y_arr: np.ndarray) -> int:
    if x_arr.shape == y_arr.shape:
        for xv, yv in zip(x_arr.flatten(), y_arr.flatten()):
            cmp_val = _cmp_scalar_with_tol(float(xv), float(yv))
            if cmp_val != 0:
                return cmp_val
        return 0
    try:
        diff = x_arr - y_arr
    except Exception:
        return (x_arr.size > y_arr.size) - (x_arr.size < y_arr.size)

    finite_parts = []
    if np.isfinite(x_arr).any():
        finite_parts.append(np.abs(x_arr[np.isfinite(x_arr)]))
    if np.isfinite(y_arr).any():
        finite_parts.append(np.abs(y_arr[np.isfinite(y_arr)]))
    scale = float(np.max(np.concatenate(finite_parts))) if finite_parts else 0.0
    tol = max(FLOAT_ATOL, FLOAT_RTOL * scale)
    for val in diff.flatten():
        if not np.isfinite(val):
            return (val > 0) - (val < 0)
        if abs(float(val)) > tol:
            return 1 if val > 0 else -1
    return 0

class NumericBase:
    """Abstract arithmetic interface used by ``group.GroupFromNumeric``."""

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
#  NumericInt
# ==========================================================
class NumericInt(NumericBase):
    """Standard integer arithmetic."""
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
# NumericFloat
# ==========================================================
class NumericFloat(NumericBase):
    """Floating-point arithmetic implemented via NumPy helpers."""
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
        if isinstance(x, (list, np.ndarray)) or isinstance(y, (list, np.ndarray)):
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            return _cmp_ndarray_with_tol(x_arr, y_arr)

        x_val, y_val = float(x), float(y)
        return _cmp_scalar_with_tol(x_val, y_val)

    def of_int(self, x): return float(x)
    def of_string(self, s): return float(s)
    def to_string(self, x): return str(float(x))


# ==========================================================
#  NumericBigInt
# ==========================================================
class NumericBigInt(NumericBase):
    """Arbitrary-precision integer arithmetic relying on Python ``int``."""
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
# NumericBigRat
# ==========================================================
class NumericBigRat(NumericBase):
    """Exact rational arithmetic based on ``fractions.Fraction``."""
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
    def compare(self, x, y): 
        return (x > y) - (x < y)

    def of_int(self, x): return Fraction(x, 1)
    def of_string(self, s):
        s = s.strip()
        if "/" in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den))
        try:
            return Fraction(s)
        except ValueError:
            return Fraction(Decimal(s))
    def to_string(self, x): return str(x)

# ==========================================================
# Tropical semiring: min-plus algebra
# ==========================================================
class TropicalNumericMinPlus(NumericBase):
    """Tropical (min,+) semiring where ⊕=min, ⊗=+, zero=+∞, one=0."""


    zero = np.inf  # tropical additive identity
    one = 0.0      # multiplicative identity

    def max(self, x, y): 
        return np.maximum(x, y)

    def min(self, x, y):
        return np.minimum(x, y)

    def add(self, x, y):
        return np.minimum(x, y)

    def neg(self, x):
        return -x

    def mul(self, x, y):
        if (np.isposinf(x) and np.isneginf(y)) or (np.isposinf(y) and np.isneginf(x)):
            return np.inf
        return np.add(x, y)

    def div(self, x, y):
        return np.subtract(x, y)

    def pow(self, x, n):
        return np.add.reduce([x] * n)

    def compare(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            
            if x_arr.shape == y_arr.shape:
                x_flat = x_arr.flatten()
                y_flat = y_arr.flatten()
                
                for i in range(len(x_flat)):
                    if x_flat[i] > y_flat[i]:
                        return 1
                    elif x_flat[i] < y_flat[i]:
                        return -1
                return 0 
            else:
                # Different shapes: compare by broadcasting rules
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


class TropicalNumericMinPlusBigRat(NumericBase):
    """Tropical (min,+) semiring backed by Fractions; zero=+inf, one=0."""

    zero = float("inf")
    one = Fraction(0, 1)

    @staticmethod
    def _is_pos_inf(value: Any) -> bool:
        return isinstance(value, (float, np.floating)) and np.isposinf(value)

    @staticmethod
    def _is_neg_inf(value: Any) -> bool:
        return isinstance(value, (float, np.floating)) and np.isneginf(value)

    def _coerce_fraction(self, value: Any) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value), 1)
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                raise TypeError("Infinity is handled separately in tropical rational arithmetic")
            return Fraction(Decimal(str(float(value))))
        raise TypeError(f"Unsupported value for tropical rational arithmetic: {value!r}")

    def _handle_order(self, x, y, prefer_min: bool):
        cmp = self.compare(x, y)
        return x if (cmp <= 0 if prefer_min else cmp >= 0) else y

    def max(self, x, y):
        return self._handle_order(x, y, prefer_min=False)

    def min(self, x, y):
        return self._handle_order(x, y, prefer_min=True)

    def add(self, x, y):
        return self.min(x, y)

    def neg(self, x):
        if self._is_pos_inf(x): 
            return float("-inf")
        if self._is_neg_inf(x): 
            return float("inf")
        return -self._coerce_fraction(x)

    def mul(self, x, y):
        if (self._is_pos_inf(x) and self._is_neg_inf(y)) or (self._is_pos_inf(y) and self._is_neg_inf(x)):
            raise ValueError("Indeterminate form (+inf) + (-inf) in tropical multiplication")
        if self._is_pos_inf(x) or self._is_pos_inf(y):
            return float("inf")
        if self._is_neg_inf(x) or self._is_neg_inf(y):
            return float("-inf")
        return self._coerce_fraction(x) + self._coerce_fraction(y)

    def div(self, x, y):
        if self._is_pos_inf(x) and self._is_pos_inf(y):
            raise ValueError("Indeterminate form inf - inf in tropical division")
        if self._is_neg_inf(x) and self._is_neg_inf(y):
            raise ValueError("Indeterminate form (-inf) - (-inf) in tropical division")
        if self._is_pos_inf(x) or self._is_neg_inf(y):
            return float("inf")
        if self._is_neg_inf(x) or self._is_pos_inf(y):
            return float("-inf")
        return self._coerce_fraction(x) - self._coerce_fraction(y)

    def pow(self, x, n):
        if n < 0:
            raise ValueError("Negative exponents are not supported in tropical min-plus")
        if n == 0:
            return self.one
        result = x
        for _ in range(1, n):
            result = self.mul(result, x)
        return result

    def compare(self, x, y) -> int:
        if self._is_pos_inf(x):
            return 0 if self._is_pos_inf(y) else 1
        if self._is_pos_inf(y):
            return -1
        if self._is_neg_inf(x):
            return 0 if self._is_neg_inf(y) else -1
        if self._is_neg_inf(y):
            return 1
        x_frac = self._coerce_fraction(x)
        y_frac = self._coerce_fraction(y)
        return (x_frac > y_frac) - (x_frac < y_frac)

    def of_int(self, x):
        return Fraction(int(x), 1)

    def of_string(self, s):
        s = s.strip()
        lower = s.lower()
        if lower in {"inf", "+inf", "+infinity", "infinity"}:
            return float("inf")
        if lower in {"-inf", "-infinity", "neg_inf"}:
            return float("-inf")
        if "/" in s:
            num, den = s.split("/", 1)
            return Fraction(int(num), int(den))
        try:
            return Fraction(s)
        except ValueError:
            return Fraction(Decimal(s))

    def to_string(self, x):
        if self._is_pos_inf(x):
            return "inf"
        if self._is_neg_inf(x):
            return "-inf"
        return str(self._coerce_fraction(x))

# ==========================================================
# Tropical semiring: max-plus algebra
# ==========================================================

class TropicalNumericMaxPlus(NumericBase):
    """Tropical (max,+) semiring where ⊕=max, ⊗=+, zero=-∞, one=0."""

    zero = -np.inf  
    one = 0.0 

    def max(self, x, y):
        return np.maximum(x, y)

    def min(self, x, y):
        return np.minimum(x, y)

    def add(self, x, y):
        return np.maximum(x, y)

    def neg(self, x):
        return -x

    def mul(self, x, y):
        if (np.isposinf(x) and np.isneginf(y)) or (np.isposinf(y) and np.isneginf(x)):
            return np.inf
        return np.add(x, y)

    def div(self, x, y):
        return np.subtract(x, y)

    def pow(self, x, n):
        return np.add.reduce([x] * n)

    def compare(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            
            if x_arr.shape == y_arr.shape:
                x_flat = x_arr.flatten()
                y_flat = y_arr.flatten()
                
                for i in range(len(x_flat)):
                    if x_flat[i] > y_flat[i]:
                        return 1
                    elif x_flat[i] < y_flat[i]:
                        return -1
                return 0 
            else:
                # Different shapes: compare by broadcasting rules
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




class TropicalNumericMaxPlusBigRat(NumericBase):
    """Tropical (max,+) semiring using Exact Rationals (Fractions).
    Zero is -inf, One is 0.
    """

    zero = float("-inf")
    one = Fraction(0, 1)

    @staticmethod
    def _is_pos_inf(value: Any) -> bool:
        return isinstance(value, (float, np.floating)) and np.isposinf(value)

    @staticmethod
    def _is_neg_inf(value: Any) -> bool:
        return isinstance(value, (float, np.floating)) and np.isneginf(value)

    def _coerce_fraction(self, value: Any) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value), 1)
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                raise TypeError("Infinity is handled separately in tropical rational arithmetic")
            return Fraction(Decimal(str(float(value))))
        raise TypeError(f"Unsupported value for tropical rational arithmetic: {value!r}")

    def _handle_inf_max(self, x, y, prefer_max: bool):
        cmp = self.compare(x, y)
        return x if (cmp >= 0 if prefer_max else cmp <= 0) else y

    def max(self, x, y):
        return self._handle_inf_max(x, y, True)

    def min(self, x, y):
        return self._handle_inf_max(x, y, False)

    def add(self, x, y):
        return self.max(x, y)

    def neg(self, x):
        if self._is_pos_inf(x):
            return float("-inf")
        if self._is_neg_inf(x):
            return float("inf")
        frac = self._coerce_fraction(x)
        return -frac

    def mul(self, x, y):
        if (self._is_pos_inf(x) and self._is_neg_inf(y)) or (self._is_pos_inf(y) and self._is_neg_inf(x)):
            raise ValueError("Indeterminate form (+inf) + (-inf) in tropical multiplication")
        if self._is_neg_inf(x) or self._is_neg_inf(y):
            return float("-inf")
        if self._is_pos_inf(x) or self._is_pos_inf(y):
            return float("inf")
        x_frac = self._coerce_fraction(x)
        y_frac = self._coerce_fraction(y)
        return x_frac + y_frac

    def div(self, x, y):
        if self._is_pos_inf(x) and self._is_pos_inf(y):
            raise ValueError("Indeterminate form inf - inf in tropical division")
        if self._is_neg_inf(x) and self._is_neg_inf(y):
            raise ValueError("Indeterminate form (-inf) - (-inf) in tropical division")
        if self._is_pos_inf(x) or self._is_neg_inf(y):
            return float("inf")
        if self._is_neg_inf(x) or self._is_pos_inf(y):
            return float("-inf")
        x_frac = self._coerce_fraction(x)
        y_frac = self._coerce_fraction(y)
        return x_frac - y_frac 

    def pow(self, x, n):
        if n < 0:
            raise ValueError("Negative exponents are not supported in tropical max-plus")
        if n == 0:
            return self.one
        result = x
        for _ in range(1, n):
            result = self.mul(result, x)
        return result

    def compare(self, x, y) -> int:
        if self._is_pos_inf(x):
            return 0 if self._is_pos_inf(y) else 1
        if self._is_pos_inf(y):
            return -1
        if self._is_neg_inf(x):
            return 0 if self._is_neg_inf(y) else -1
        if self._is_neg_inf(y):
            return 1
        x_frac = self._coerce_fraction(x)
        y_frac = self._coerce_fraction(y)
        if isinstance(x_frac, Fraction) and isinstance(y_frac, Fraction):
            return (x_frac > y_frac) - (x_frac < y_frac)
        raise TypeError("Comparison expects finite tropical rational values or infinities")

    def of_int(self, x):
        return Fraction(int(x), 1)

    def of_string(self, s):
        s = s.strip()
        lower = s.lower()
        if lower in {"inf", "+inf", "+infinity", "infinity"}:
            return float("inf")
        if lower in {"-inf", "-infinity", "neg_inf"}:
            return float("-inf")
        if "/" in s:
            num, den = s.split("/", 1)
            return Fraction(int(num), int(den))
        try:
            return Fraction(s)
        except ValueError:
            return Fraction(Decimal(s))

    def to_string(self, x):
        if self._is_pos_inf(x):
            return "inf"
        if self._is_neg_inf(x):
            return "-inf"
        return str(self._coerce_fraction(x))


# ==========================================================
#  Registry -- Get() interface
# ==========================================================
NUMERIC_MODULES: Dict[str, Type[NumericBase]] = {
    "Numeric_int": NumericInt,
    "Numeric_float": NumericFloat,
    "Numeric_big_int": NumericBigInt,
    "Numeric_big_rat": NumericBigRat,
    "tropical_min_plus": TropicalNumericMinPlus,
    "tropical_min_plus_big_rat": TropicalNumericMinPlusBigRat,
    "tropical_max_plus": TropicalNumericMaxPlus,
    "tropical_max_plus_big_rat": TropicalNumericMaxPlusBigRat,
}


def get(name: str) -> NumericBase:
    try:
        return NUMERIC_MODULES[name]()
    except KeyError:
        raise ValueError(f"{name}: unknown type of numerical data")


def get_name_of_modules():
    return list(NUMERIC_MODULES.keys())
