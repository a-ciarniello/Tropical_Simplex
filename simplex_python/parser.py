"""Recursive-descent parser for the custom tropical LP format."""

import re
import numeric
import linear_prog
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterator

TOKEN_SPEC = [
    ("LEQ", r"<="),
    ("GEQ", r">="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("EQ", r"="),
    ("NUM", r"(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SKIP", r"[ \t]+"),
    ("NEWLINE", r"\n"),
    ("MISMATCH", r"."),
]


@dataclass
class Token:
    """Single lexical token emitted by ``tokenize``."""

    typ: str
    value: str


class ParseError(Exception):
    """Canonical parser error for tropical LP inputs."""


def tokenize(text: str) -> Iterator[Token]:
    """Yield tokens from ``text`` according to ``TOKEN_SPEC``."""
    regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC)
    for mo in re.finditer(regex, text):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "SKIP" or kind == "NEWLINE":
            continue
        if kind == "IDENT":
            lw = value.lower()
            if lw == "max":
                yield Token("MAX", value)
                continue
            if lw == "min":
                yield Token("MIN", value)
                continue
            yield Token("VAR", value)
        elif kind == "NUM":
            yield Token(kind, value)
        elif kind == "MISMATCH":
            raise ParseError(f"Unexpected character: {value}")
        else:
            yield Token(kind, value)


# ------------------------
# Recursive Descent for linear forms
# ------------------------

class RDParser:
    """Recursive-descent parser for linear forms under the tropical grammar."""
    def __init__(
        self,
        tokens: List[Token],
        var_names: Dict[str, int],
        semiring: str,
        numeric_module: Optional[numeric.NumericBase],
    ):
        self.tokens = tokens
        self.pos = 0
        self.var_names = var_names
        self.semiring = semiring
        self.Num = numeric_module if numeric_module is not None else numeric.NumericFloat()

    def _zero(self):
        zero = self.Num.zero
        return zero() if callable(zero) else zero

    def _literal(self, value: str):
        return self.Num.of_string(value)

    def _negate_terms(self, terms: List[Tuple[linear_prog.ColIndex, Any]]):
        return [(idx, self.Num.neg(coef)) for idx, coef in terms]

    def _peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]  
        else: 
            return None

    def _eat(self, *expected: str) -> Token:
        tok = self._peek()
        if tok is None or tok.typ not in expected:
            exp = " or ".join(expected)
            raise ParseError(f"Expected {exp}, got {tok}")
        self.pos += 1
        return tok

    def parse_linear_form(self) -> List[Tuple[linear_prog.ColIndex, float]]:
        """Parse a full linear form and return a flat list of terms."""
        terms = self.parse_sum()
        return terms

    def parse_sum(self) -> List[Tuple[linear_prog.ColIndex, Any]]:
        """Handle chained additions/subtractions within a linear form."""
        negate = False
        tok = self._peek()
        if tok and tok.typ in ("PLUS", "MINUS"):
            self._eat(tok.typ)
            negate = tok.typ == "MINUS"

        terms = self.parse_term()
        if negate:
            terms = self._negate_terms(terms)

        while True:
            tok = self._peek()
            if tok and tok.typ in ("PLUS", "MINUS"):
                self._eat(tok.typ)
                rhs = self.parse_term()
                if tok.typ == "MINUS":
                    rhs = [(idx, -coef) for idx, coef in rhs]
                terms.extend(rhs)
            else:
                break
        return terms

    def parse_term(self) -> List[Tuple[linear_prog.ColIndex, Any]]:
        """Parse elementary terms (scalars, variables, or min/max blocks)."""
        negate = False
        while True:
            tok = self._peek()
            if tok and tok.typ in ("PLUS", "MINUS"):
                self._eat(tok.typ)
                if tok.typ == "MINUS":
                    negate = not negate
            else:
                break

        if tok is None:
            raise ParseError("Unexpected end of expression")

        if tok.typ == "NUM":
            self._eat("NUM")
            coeff = self._literal(tok.value)
            
            if self._peek() and self._peek().typ == "MUL":
                self._eat("MUL")
                var_tok = self._eat("VAR")
                if var_tok.value not in self.var_names:
                    raise ParseError(f"Unknown variable '{var_tok.value}'")
                idx = self.var_names[var_tok.value]
                terms = [((linear_prog.ColKind.VAR, idx), coeff)]
            else:
                terms = [((linear_prog.ColKind.AFFINE, None), coeff)]
            if negate:
                terms = self._negate_terms(terms)
            return terms

        if tok.typ == "VAR":
            self._eat("VAR")
            if tok.value not in self.var_names:
                raise ParseError(f"Unknown variable '{tok.value}'")
            j = self.var_names[tok.value]
            coeff = self._zero()
            while True:
                nxt = self._peek()
                if nxt and nxt.typ in ("PLUS", "MINUS"):
                    op = nxt.typ
                    self._eat(nxt.typ)
                    num_tok = self._eat("NUM")
                    delta = self._literal(num_tok.value)
                    if op == "MINUS":
                        delta = self.Num.neg(delta)
                    coeff = self.Num.add(coeff, delta)
                else:
                    break
            terms = [((linear_prog.ColKind.VAR, j), coeff)]
            if negate:
                terms = self._negate_terms(terms)
            return terms

        if tok.typ in ("MAX", "MIN"):
            fn = tok.typ
            # Hybrid objective/constraints: allow both max and min even if semiring is minplus or maxplus.
            self._eat(fn)
            self._eat("LPAREN")
            all_terms: List[Tuple[linear_prog.ColIndex, float]] = []
            all_terms.extend(self.parse_sum())

            while True:
                tok = self._peek()
                if tok and tok.typ == "COMMA":
                    self._eat("COMMA")
                    all_terms.extend(self.parse_sum())
                else:
                    break

            self._eat("RPAREN")
            if negate:
                all_terms = self._negate_terms(all_terms)
            return all_terms

        if tok.typ == "LPAREN":
            self._eat("LPAREN")
            inner = self.parse_sum()
            self._eat("RPAREN")
            if negate:
                inner = self._negate_terms(inner)
            return inner

        raise ParseError(f"Unexpected token {tok}")


# ------------------------
# High level parsing
# ------------------------


def parse_linear_form_str(
    expr: str,
    var_names: Dict[str, int],
    semiring: str,
    num_module: Optional[numeric.NumericBase],
) -> List[Tuple[linear_prog.ColIndex, Any]]:
    """Tokenize and parse ``expr`` into a list of column/weight pairs."""
    tokens = list(tokenize(expr))
    parser = RDParser(tokens, var_names, semiring, num_module)
    terms = parser.parse_linear_form()


    if parser._peek() is not None:
        raise ParseError(f"Unexpected trailing token {parser._peek()}")
    
    return terms


def parse_basic_point_line(line: str, num_module: Optional[numeric.NumericBase]) -> List[Any]:
    """Extract the basic-point coordinates from a ``basic point = ...`` line."""
    if '=' in line:
        values_part = line.split('=', 1)[1].strip().rstrip(';')
        if not values_part:
            return []
        Num = num_module if num_module is not None else numeric.NumericFloat()
        return [Num.of_string(v.strip()) for v in values_part.split(',') if v.strip()]
    return []


def map_numeric_to_module_name(numeric_type: str, semiring: Optional[str]) -> str:

    numeric_lc = numeric_type.strip().lower()
    mapping = {
        "int": "Numeric_int",
        "float": "Numeric_float",
        "big_int": "Numeric_big_int",
        "big_rat": "Numeric_big_rat",
        "tropical_min_plus": "tropical_min_plus",
        "tropical_max_plus": "tropical_max_plus",
    }

    if numeric_lc in mapping:
        return mapping[numeric_lc]

    if numeric_lc in numeric.NUMERIC_MODULES:
        return numeric_lc

    raise ParseError(f"Unknown numeric type '{numeric_type}'")


def parse_objective_line(
    line: str,
    var_names: Dict[str, int],
    semiring: str,
    num_module: Optional[numeric.NumericBase],
) -> Tuple[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]], bool]:
    is_maximize = line.startswith("maximize")
    if is_maximize:
        expr = line[len("maximize"):]
    else:
        expr = line[len("minimize"):]

    expr = expr.strip().rstrip(';')
    terms = parse_linear_form_str(expr, var_names, semiring, num_module)

    if is_maximize:
        sign = linear_prog.Sign.NEG
    else:
        sign = linear_prog.Sign.POS
    
    obj = [(col_idx, sign, coeff) for col_idx, coeff in terms]
    return obj, is_maximize


def parse_inequality_line(
    line: str,
    var_names: Dict[str, int],
    semiring: str,
    num_module: Optional[numeric.NumericBase],
) -> List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]]:
    line = line.rstrip(';').strip()
    if "<=" in line:
        left, right = line.split("<=", 1)
        op = "LEQ"
    elif ">=" in line:
        left, right = line.split(">=", 1)
        op = "GEQ"
    else:
        raise ParseError("Inequality must contain <= or >=")



    left_terms = parse_linear_form_str(left.strip(), var_names, semiring, num_module)
    right_terms = parse_linear_form_str(right.strip(), var_names, semiring, num_module)


    result: List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]] = []
    if op == "LEQ":

        result.extend((col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in left_terms)
        result.extend((col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in right_terms)
    else:

        result.extend((col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in left_terms)
        result.extend((col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in right_terms)
    return result


def parse_lp(content: str, num_module: Optional[numeric.NumericBase] = None) -> Tuple[str, Dict[str, int], Any, Any, Any, List[Any], bool]:
    lines = content.strip().split('\n')

    state = "header"
    var_names: Dict[str, int] = {}
    semiring: Optional[str] = None
    numeric_type = "float"
    objective = None
    is_maximize = False
    ineqs: List[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, Any]]] = []
    basic_point: List[Any] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        if state == "header":
            if line.startswith("vars:"):
                vars_part = line[len("vars:"):].strip()
                var_list = [v.strip() for v in vars_part.split(',') if v.strip()]
                for idx, var in enumerate(var_list):
                    var_names[var] = idx
            elif line.startswith("semiring:"):
                semiring = line[len("semiring:"):].strip()
            elif line.startswith("numeric:"):
                numeric_type = line[len("numeric:"):].strip()
            elif line.startswith("lp:"):
                state = "lp"
        elif state == "lp":
            if line.startswith("maximize") or line.startswith("minimize"):
                if semiring is None:
                    if line.startswith("maximize"):
                        semiring = "maxplus"
                    else:
                        semiring = "minplus"
                obj, is_max = parse_objective_line(line, var_names, semiring, num_module)
                objective = obj
                is_maximize = is_max
            elif line.startswith("basic point"):
                basic_point = parse_basic_point_line(line, num_module)
            elif "<=" in line or ">=" in line:
                if semiring is None:
                    semiring = "minplus"  # default fallback
                ineqs.append(parse_inequality_line(line, var_names, semiring, num_module))

    if semiring is None:
        semiring = "minplus"

    numeric_name = map_numeric_to_module_name(numeric_type, semiring)

    return numeric_name, var_names, semiring, objective, ineqs, basic_point, is_maximize


class Parser:
    """Parser for tropical LP files"""

    def __init__(self, Num: numeric.NumericBase):
        self.Num = Num

    def main(self, content: str) -> Tuple[Any, Any, np.ndarray, bool]:
        numeric_name, var_names, semiring, objective, ineqs, basic_point, is_maximize = parse_lp(content, self.Num)

        zero_attr = getattr(self.Num, "zero", 0.0)
        zero_entry = zero_attr() if callable(zero_attr) else zero_attr
        obj_formatted = objective if objective else [((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, zero_entry)]
        return obj_formatted, ineqs, np.array(basic_point, dtype=object), is_maximize


def lexer_from_file(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def lexer_header(text: str) -> Tuple[str, Dict[str, int], Any]:
    numeric_name, var_names, semiring, _, _, _, _ = parse_lp(text)
    return numeric_name, var_names, semiring