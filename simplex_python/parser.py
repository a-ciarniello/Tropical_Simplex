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

def parse_simple_lp(content: str) -> Tuple[str, Dict[str, int], Any, Any, List[float], bool]:
    """
    Parser semplificato per file LP con formato:
    vars: x, y, z
    semiring: minplus 
    numeric: float
    lp:
    maximize x + 3;
    x + 2 <= y + 4;
    basic point = 1.0, 2.0;
    
    Returns:
        Tuple containing:
        - numeric_name: str (e.g., "tropical_min_plus")
        - var_names: Dict[str, int] (variable name to index mapping)
        - objective: objective function terms
        - ineqs: list of inequalities
        - basic_point: list of floats (optional basic point)
        - is_maximize: bool (True if maximize, False if minimize)
    """
    lines = content.strip().split('\n')
    
    # State parsing
    state = "header"
    var_names = {}
    semiring = "maxplus"
    numeric_type = "float"
    objective = None
    is_maximize = False  # Track if objective is maximize or minimize
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
                is_maximize = line.startswith("maximize")  # Track the objective type
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
    
    return numeric_name, var_names, objective, ineqs, basic_point, is_maximize

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
    """
    Parse simple expression like 'x + 2' or 'y + 4' or just '1'
    Returns list of (col_index, coefficient) tuples
    """
    terms = []
    
    # Remove whitespace for easier parsing
    expr = expr.replace(' ', '')
    
    # Check if expression contains any variables
    found_var = False
    for var_name, var_idx in var_names.items():
        if var_name in expr:
            found_var = True
            col_index = (linear_prog.ColKind.VAR, var_idx)
            # Variable always has coefficient 0.0 in tropical LP format
            # The actual value comes from the point, not the coefficient
            terms.append((col_index, 0.0))
            # Remove the variable from expression to find constants
            expr = expr.replace(var_name, '')
            break
    
    # Clean up the remaining expression (remove operators without operands)
    expr = expr.strip('+-')
    
    # Look for constants in the remaining expression
    if expr:
        try:
            # Try to parse as a number
            const_val = float(expr)
            if const_val != 0 or not found_var:  # Always include constant if no variable, or if non-zero
                col_index = (linear_prog.ColKind.AFFINE, None)
                terms.append((col_index, const_val))
        except ValueError:
            # Not a simple number, might have multiple terms
            # Extract all numbers
            numbers = re.findall(r'[+-]?\d+\.?\d*', expr)
            if numbers:
                const_val = sum(float(n) for n in numbers)
                if const_val != 0 or not found_var:
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
        
    def main(self, content: str) -> Tuple[Any, Any, np.ndarray, bool]:
        """
        Metodo principale per parsing del file LP
        
        Returns:
            Tuple containing:
            - objective: formatted objective function
            - ineqs: list of inequalities
            - basic_point: numpy array of basic point
            - is_maximize: bool (True if maximize, False if minimize)
        """
        numeric_name, var_names, objective, ineqs, basic_point, is_maximize = parse_simple_lp(content)
        
        # Convert to proper format
        if objective:
            obj_formatted = objective
        else:
            # Default objective if none specified
            obj_formatted = [((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, 0.0)]
            
        return obj_formatted, ineqs, np.array(basic_point), is_maximize

def lexer_from_file(fp: str) -> str:
    """Legge il file e restituisce il contenuto"""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()

def lexer_header(text: str) -> Tuple[str, Dict[str, int]]:
    """Parsing solo dell'header per ottenere numeric_name e var_names"""
    numeric_name, var_names, _, _, _, _ = parse_simple_lp(text)
    return numeric_name, var_names