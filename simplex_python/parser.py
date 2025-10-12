"""
Parser semplificato per i file LP in formato simplex tropicale
Versione più semplice e funzionale
"""

import re
import numeric
import linear_prog
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class ParseError(Exception):
    pass

def parse_simple_lp(content: str) -> Tuple[str, Dict[str, int], Any, Any, List[float]]:
    """
    Parser semplificato per file LP con formato:
    vars: x, y, z
    semiring: minplus 
    numeric: float
    lp:
    maximize x + 3;
    x + 2 <= y + 4;
    basic point = 1.0, 2.0;
    """
    lines = content.strip().split('\n')
    
    # State parsing
    state = "header"
    var_names = {}
    semiring = "maxplus"
    numeric_type = "float"
    objective = None
    ineqs = []
    basic_point = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue
            
        if state == "header":
            if line.startswith("vars:"):
                # Parse vars: x, y, z
                vars_part = line[5:].strip()
                var_list = [v.strip() for v in vars_part.split(',')]
                for idx, var in enumerate(var_list):
                    var_names[var] = idx
            elif line.startswith("semiring:"):
                semiring = line[9:].strip()
            elif line.startswith("numeric:"):
                numeric_type = line[8:].strip()
            elif line.startswith("lp:"):
                state = "lp"
        
        elif state == "lp":
            if line.startswith("maximize") or line.startswith("minimize"):
                objective = parse_objective_line(line, var_names)
            elif line.startswith("basic point"):
                basic_point = parse_basic_point_line(line)
            elif any(op in line for op in ["<=", ">="]):
                ineq = parse_inequality_line(line, var_names, semiring)
                if ineq:
                    ineqs.append(ineq)
        
        i += 1
    
    # Convert semiring to numeric name
    if semiring == "minplus":
        numeric_name = "tropical_min_plus"
    else:
        numeric_name = "tropical_max_plus"
    
    return numeric_name, var_names, objective, ineqs, basic_point

def parse_objective_line(line: str, var_names: Dict[str, int]) -> List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]]:
    """Parse maximize/minimize line"""
    if line.startswith("maximize"):
        expr = line[8:].strip().rstrip(';')
        sign = linear_prog.Sign.NEG  # maximize -> negative coefficients
    else:  # minimize
        expr = line[8:].strip().rstrip(';')
        sign = linear_prog.Sign.POS  # minimize -> positive coefficients
    
    # Simple parsing: assume format "var + constant" or "constant"
    terms = []
    
    # Very basic parsing - handle simple cases
    if any(var in expr for var in var_names):
        # Find variables in expression
        for var_name, var_idx in var_names.items():
            if var_name in expr:
                col_index = (linear_prog.ColKind.VAR, var_idx)
                # Extract coefficient (simplified)
                coeff = 0.0
                if '+' in expr:
                    parts = expr.split('+')
                    for part in parts:
                        if var_name not in part and part.strip().replace('.','').replace('-','').isdigit():
                            coeff = float(part.strip())
                terms.append((col_index, sign, coeff))
                break
    
    # Add constant term if present
    numbers = re.findall(r'[-+]?\\d*\\.?\\d+', expr)
    if numbers:
        const_val = float(numbers[0])
        col_index = (linear_prog.ColKind.AFFINE, None)
        terms.append((col_index, sign, const_val))
    
    return terms

def parse_inequality_line(line: str, var_names: Dict[str, int], semiring: str) -> Optional[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]]]:
    """Parse inequality line like 'x + 2 <= y + 4;'"""
    line = line.rstrip(';').strip()
    
    if "<=" in line:
        left, right = line.split("<=")
        op = "<="
    elif ">=" in line:
        left, right = line.split(">=")  
        op = ">="
    else:
        return None
    
    left = left.strip()
    right = right.strip()
    
    # Parse left and right sides
    left_terms = parse_expression(left, var_names)
    right_terms = parse_expression(right, var_names)
    
    # Convert to LP format: left <= right becomes right - left >= 0
    result = []
    
    if op == "<=":
        # Adjust for semiring
        if semiring == "minplus":
            # In minplus: <= becomes >=
            result.extend([(col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in left_terms])
            result.extend([(col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in right_terms])
        else:
            # In maxplus: <= stays <=
            result.extend([(col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in right_terms])
            result.extend([(col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in left_terms])
    else:  # ">="
        if semiring == "minplus":
            # In minplus: >= becomes <=
            result.extend([(col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in right_terms])
            result.extend([(col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in left_terms])
        else:
            # In maxplus: >= stays >=
            result.extend([(col_idx, linear_prog.Sign.POS, coeff) for col_idx, coeff in left_terms])
            result.extend([(col_idx, linear_prog.Sign.NEG, coeff) for col_idx, coeff in right_terms])
    
    return result

def parse_expression(expr: str, var_names: Dict[str, int]) -> List[Tuple[linear_prog.ColIndex, float]]:
    """Parse simple expression like 'x + 2' or 'y + 4'"""
    terms = []
    
    # Very simple parsing
    expr = expr.replace(' ', '')
    
    # Look for variables
    for var_name, var_idx in var_names.items():
        if var_name in expr:
            col_index = (linear_prog.ColKind.VAR, var_idx)
            # Simple coefficient extraction
            coeff = 0.0
            
            # Pattern: var + number or var - number
            pattern = f'{var_name}\\s*([+-])\\s*(\\d+(?:\\.\\d+)?)'
            match = re.search(pattern, expr)
            if match:
                sign, num = match.groups()
                coeff = float(num) if sign == '+' else -float(num)
            
            terms.append((col_index, coeff))
            # Remove this variable from expression
            expr = re.sub(f'{var_name}\\s*[+-]?\\s*\\d*\\.?\\d*', '', expr)
            break
    
    # Look for standalone numbers (constants)
    numbers = re.findall(r'[+-]?\\d*\\.?\\d+', expr)
    if numbers:
        const_val = sum(float(n) for n in numbers)
        if const_val != 0:
            col_index = (linear_prog.ColKind.AFFINE, None)
            terms.append((col_index, const_val))
    
    return terms

def parse_basic_point_line(line: str) -> List[float]:
    """Parse basic point line like 'basic point = 1.0, 2.0, 3.0;'"""
    # Extract values after '='
    if '=' in line:
        values_part = line.split('=')[1].strip().rstrip(';')
        values = [float(v.strip()) for v in values_part.split(',')]
        return values
    return []

class Parser:
    """Parser semplificato per i file LP tropicali"""
    
    def __init__(self, Num: numeric.NumericBase):
        self.Num = Num
        
    def main(self, content: str) -> Tuple[Any, Any, np.ndarray]:
        """Metodo principale per parsing del file LP"""
        numeric_name, var_names, objective, ineqs, basic_point = parse_simple_lp(content)
        
        # Convert to proper format
        if objective:
            obj_formatted = objective
        else:
            # Default objective if none specified
            obj_formatted = [((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 0.0)]
            
        return obj_formatted, ineqs, np.array(basic_point)

def lexer_from_file(fp: str) -> str:
    """Legge il file e restituisce il contenuto"""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()

def lexer_header(text: str) -> Tuple[str, Dict[str, int]]:
    """Parsing solo dell'header per ottenere numeric_name e var_names"""
    numeric_name, var_names, _, _, _ = parse_simple_lp(text)
    return numeric_name, var_names