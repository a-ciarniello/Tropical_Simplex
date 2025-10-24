"""
Simplified parser for tropical simplex LP files
Simplified and functional version
"""

import re
import numeric
import linear_prog
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class ParseError(Exception):
    pass

def parse_max_expression(expr: str, var_names: Dict[str, int]) -> List[Tuple[linear_prog.ColIndex, float]]:
    """
    Parse expression that may contain max() or min() function.
    
    Examples:
        max(x, y-4) -> [(VAR x, 0.0), (VAR y, -4.0)]
        min(x, y-4) -> [(VAR x, 0.0), (VAR y, -4.0)]
        x + 3 -> [(VAR x, 0.0), (AFFINE, 3.0)]
        max(x, y) -> [(VAR x, 0.0), (VAR y, 0.0)]
    
    Returns:
        List of (col_index, coefficient) tuples representing tropical sum (max/min in standard algebra)
    
    Note: Both max() and min() are parsed the same way in tropical algebra representation.
          The interpretation depends on the semiring:
          - In minplus semiring: ⊕ = min, so max() in LP represents tropical addition
          - In maxplus semiring: ⊕ = max, so min() in LP represents tropical addition
    """
    expr = expr.strip()
    
    # Check if expression starts with max(...) or min(...)
    func_name = None
    if expr.startswith('max('):
        func_name = 'max'
        start_idx = 4
    elif expr.startswith('min('):
        func_name = 'min'
        start_idx = 4
    
    if func_name:
        # Find matching closing parenthesis
        paren_count = 0
        end_idx = -1
        for i, c in enumerate(expr):
            if c == '(':
                paren_count += 1
            elif c == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            raise ParseError(f"Unmatched parentheses in {func_name} expression: {expr}")
        
        # Extract content inside max(...) or min(...)
        func_content = expr[start_idx:end_idx]
        
        # Split by comma (careful with nested max/min)
        args = []
        current_arg = ""
        paren_depth = 0
        
        for c in func_content:
            if c == ',' and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if c == '(':
                    paren_depth += 1
                elif c == ')':
                    paren_depth -= 1
                current_arg += c
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Parse each argument recursively
        all_terms = []
        for arg in args:
            arg_terms = parse_simple_expression(arg, var_names)
            all_terms.extend(arg_terms)
        
        return all_terms
    else:
        # Simple expression without max() or min()
        return parse_simple_expression(expr, var_names)

def parse_simple_expression(expr: str, var_names: Dict[str, int]) -> List[Tuple[linear_prog.ColIndex, float]]:
    """
    Parse simple expression like 'x', 'y-4', 'x+3', '5'
    Returns list of (col_index, coefficient) tuples
    """
    expr = expr.strip()
    terms = []
    
    # Try to match pattern: variable [+-] constant
    # First, check if there's a variable
    found_var = None
    for var_name, var_idx in var_names.items():
        if var_name in expr:
            found_var = (var_name, var_idx)
            break
    
    if found_var:
        var_name, var_idx = found_var
        col_index = (linear_prog.ColKind.VAR, var_idx)
        
        # Extract coefficient after the variable
        # Pattern: var_name followed by optional [+-]number
        var_pos = expr.find(var_name)
        after_var = expr[var_pos + len(var_name):].strip()
        
        coeff = 0.0
        if after_var:
            # Remove whitespace
            after_var = after_var.replace(' ', '')
            if after_var:
                try:
                    # Try to parse as signed number
                    coeff = float(after_var)
                except ValueError:
                    # Might have format like "+3" or "-4"
                    pass
        
        terms.append((col_index, coeff))
    else:
        # Just a constant
        try:
            const_val = float(expr)
            col_index = (linear_prog.ColKind.AFFINE, None)
            terms.append((col_index, const_val))
        except ValueError:
            # Could not parse
            pass
    
    return terms

def parse_simple_lp(content: str) -> Tuple[str, Dict[str, int], Any, Any, List[float], bool]:
    """
    Simplified parser for LP files with format:
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
    semiring = None  # Will be deduced if not specified
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
                
                # Auto-deduce semiring if not specified
                # minimize -> minplus (⊕ = min)
                # maximize -> maxplus (⊕ = max)
                if semiring is None:
                    semiring = "maxplus" if is_maximize else "minplus"
                    
            elif line.startswith("basic point"):
                basic_point = parse_basic_point_line(line)
            elif any(op in line for op in ["<=", ">="]):
                ineq = parse_inequality_line(line, var_names, semiring if semiring else "minplus")
                if ineq:
                    ineqs.append(ineq)
        
        i += 1
    
    # Default to minplus if still not set
    if semiring is None:
        semiring = "minplus"
    
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
    
    # Parse the expression (handles max() function)
    expr_terms = parse_max_expression(expr, var_names)
    
    # Apply sign to all terms
    terms = [(col_idx, sign, coeff) for col_idx, coeff in expr_terms]
    
    return terms

def parse_inequality_line(line: str, var_names: Dict[str, int], semiring: str) -> Optional[List[Tuple[linear_prog.ColIndex, linear_prog.Sign, float]]]:
    """Parse inequality line like 'x + 2 <= y + 4;' or 'max(x,y) <= 3;' or 'min(x,y) >= 1;'"""
    line = line.rstrip(';').strip()
    
    if "<=" in line:
        left, right = line.split("<=", 1)
        op = "<="
    elif ">=" in line:
        left, right = line.split(">=", 1)  
        op = ">="
    else:
        return None
    
    left = left.strip()
    right = right.strip()
    
    # Parse left and right sides (handles max() expressions)
    left_terms = parse_max_expression(left, var_names)
    right_terms = parse_max_expression(right, var_names)
    
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

def parse_basic_point_line(line: str) -> List[float]:
    """Parse basic point line like 'basic point = 1.0, 2.0, 3.0;'"""
    # Extract values after '='
    if '=' in line:
        values_part = line.split('=')[1].strip().rstrip(';')
        values = [float(v.strip()) for v in values_part.split(',')]
        return values
    return []

class Parser:
    """Simplified parser for tropical LP files"""
    
    def __init__(self, Num: numeric.NumericBase):
        self.Num = Num
        
    def main(self, content: str) -> Tuple[Any, Any, np.ndarray, bool]:
        """
        Main method for LP file parsing
        
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
    """Read the file and return its content"""
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()

def lexer_header(text: str) -> Tuple[str, Dict[str, int]]:
    """Parse only the header to get numeric_name and var_names"""
    numeric_name, var_names, _, _, _, _ = parse_simple_lp(text)
    return numeric_name, var_names