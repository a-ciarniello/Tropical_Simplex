import re
import numeric
import linear_prog
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterator
import numpy as np


class ParseError(Exception):
    pass


# ------------------------
# Tokenizer
# ------------------------

TOKEN_SPEC = [
    ("LEQ", r"<="),
    ("GEQ", r">="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
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
    typ: str
    value: str


def tokenize(text: str) -> Iterator[Token]:
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
    def __init__(self, tokens: List[Token], var_names: Dict[str, int], semiring: str):
        self.tokens = tokens
        self.pos = 0
        self.var_names = var_names
        self.semiring = semiring  # "minplus" or "maxplus"

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
        terms = self.parse_sum()
        return terms

    def parse_sum(self) -> List[Tuple[linear_prog.ColIndex, float]]:
        terms = self.parse_term()
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

    def parse_term(self) -> List[Tuple[linear_prog.ColIndex, float]]:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")

        if tok.typ == "NUM":
            self._eat("NUM")
            return [((linear_prog.ColKind.AFFINE, None), float(tok.value))]

        if tok.typ == "VAR":
            self._eat("VAR")
            if tok.value not in self.var_names:
                raise ParseError(f"Unknown variable '{tok.value}'")
            j = self.var_names[tok.value]
            return [((linear_prog.ColKind.VAR, j), 0.0)]

        if tok.typ in ("MAX", "MIN"):
            fn = tok.typ
            # Enforce semiring restrictions 
            if fn == "MAX" and self.semiring == "minplus":
                raise ParseError("'max' not allowed in minplus semiring")
            if fn == "MIN" and self.semiring == "maxplus":
                raise ParseError("'min' not allowed in maxplus semiring")
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
            return all_terms

        if tok.typ == "LPAREN":
            self._eat("LPAREN")
            inner = self.parse_sum()
            self._eat("RPAREN")
            return inner

        raise ParseError(f"Unexpected token {tok}")


# ------------------------
# High level parsing (header + rows)
# ------------------------


def parse_linear_form_str(expr: str, var_names: Dict[str, int], semiring: str) -> List[Tuple[linear_prog.ColIndex, float]]:
    tokens = list(tokenize(expr))
    parser = RDParser(tokens, var_names, semiring)
    terms = parser.parse_linear_form()

    # All tokens must be consumed
    if parser._peek() is not None:
        raise ParseError(f"Unexpected trailing token {parser._peek()}")
    
    return terms


def parse_basic_point_line(line: str) -> List[float]:
    if '=' in line:
        values_part = line.split('=', 1)[1].strip().rstrip(';')
        return [float(v.strip()) for v in values_part.split(',') if v.strip()]
    return []


def map_semiring_to_numeric_name(semiring: str) -> str:
    if semiring == "minplus":
        return "tropical_min_plus"
    if semiring == "maxplus":
        return "tropical_max_plus"
    raise ParseError(f"Unknown semiring '{semiring}'")


def parse_objective_line(line: str, var_names: Dict[str, int], semiring: str) -> Tuple[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]], bool]:
    is_maximize = line.startswith("maximize")
    if is_maximize:
        expr = line[len("maximize"):]
    else:
        expr = line[len("minimize"):]

    expr = expr.strip().rstrip(';')
    terms = parse_linear_form_str(expr, var_names, semiring)

    if is_maximize:
        sign = linear_prog.Sign.NEG
    else:
        sign = linear_prog.Sign.POS
    
    obj = [(col_idx, sign, coeff) for col_idx, coeff in terms]
    return obj, is_maximize


def parse_inequality_line(line: str, var_names: Dict[str, int], semiring: str) -> List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]]:
    line = line.rstrip(';').strip()
    if "<=" in line:
        left, right = line.split("<=", 1)
        op = "LEQ"
    elif ">=" in line:
        left, right = line.split(">=", 1)
        op = "GEQ"
    else:
        raise ParseError("Inequality must contain <= or >=")



    # Lexer inversion: swap in minplus
    if semiring == "minplus":
        op = "GEQ" if op == "LEQ" else "LEQ"

    left_terms = parse_linear_form_str(left.strip(), var_names, semiring)
    right_terms = parse_linear_form_str(right.strip(), var_names, semiring)


    result: List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]] = []
    if op == "LEQ":

        # l1 <= l2  ==> Pos on right, Neg on left
        result.extend((col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in left_terms)
        result.extend((col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in right_terms)
    else:  # GEQ

        # l1 >= l2  ==> Pos on left, Neg on right
        result.extend((col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in left_terms)
        result.extend((col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in right_terms)
    return result


def parse_lp(content: str) -> Tuple[str, Dict[str, int], Any, Any, List[float], bool]:
    lines = content.strip().split('\n')

    state = "header"
    var_names: Dict[str, int] = {}
    semiring: Optional[str] = None
    numeric_type = "float"  # kept for compatibility
    objective = None
    is_maximize = False
    ineqs: List[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = []
    basic_point: List[float] = []

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
                # Deduce semiring if absent
                if semiring is None:
                    if line.startswith("maximize"):
                        semiring = "maxplus"
                    else:
                        semiring = "minplus"
                obj, is_max = parse_objective_line(line, var_names, semiring)
                objective = obj
                is_maximize = is_max
            elif line.startswith("basic point"):
                basic_point = parse_basic_point_line(line)
            elif "<=" in line or ">=" in line:
                if semiring is None:
                    semiring = "minplus"  # default fallback
                ineqs.append(parse_inequality_line(line, var_names, semiring))

    if semiring is None:
        semiring = "minplus"

    numeric_name = map_semiring_to_numeric_name(semiring)

    return numeric_name, var_names, objective, ineqs, basic_point, is_maximize


class Parser:
    """Parser for tropical LP files"""

    def __init__(self, Num: numeric.NumericBase):
        self.Num = Num

    def main(self, content: str) -> Tuple[Any, Any, np.ndarray, bool]:
        numeric_name, var_names, objective, ineqs, basic_point, is_maximize = parse_lp(content)

        obj_formatted = objective if objective else [((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 0.0)]
        return obj_formatted, ineqs, np.array(basic_point), is_maximize


def lexer_from_file(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def lexer_header(text: str) -> Tuple[str, Dict[str, int]]:
    numeric_name, var_names, _, _, _, _ = parse_lp(text)
    return numeric_name, var_names