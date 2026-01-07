from __future__ import annotations
import numpy as np
import numeric, group, linear_prog, tangent_digraph
from simplet import Simplet
from linear_prog import ColKind, RowKind, Sign
import parser

# === 1. Build the numeric field and group ===
print("Available numeric fields:")
available_fields = numeric.get_name_of_modules()
print(available_fields)

# Use "tropical_min_plus" for the tropical problem
Num = numeric.get("tropical_min_plus")
print(f"Selected numeric field: {Num}")
G = group.GroupFromNumeric(Num)
print(f"Created group: {G}")

# === 2. Load LP from file ===
LPmod = linear_prog.LinearProg(G)
print("Loading LP from file problems/test.lp...")
try:
    content = parser.lexer_from_file("problems/test.lp")
    numeric_name, var_names, objective, ineqs, basic_point, is_maximize = parser.parse_lp(content)
    
    # Create LP
    nb_vars = len(var_names)
    lp = LPmod.init(lambda j: f"x{j}", nb_vars, objective, ineqs)
    basic_point = np.array(basic_point)
    
    print(f"LP loaded successfully!")
    print(f"Variables: {var_names}")
    print(f"Basic point from file: {basic_point}")
except Exception as e:
    print(f"Error loading file: {e}")
    print("Using a simple LP...")
    
    # Simple LP: maximize x0, subject to x0 <= 2  
    objective: list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]] = [((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 1.0)]
    ineqs: list[list[tuple[linear_prog.ColIndex, linear_prog.Sign, float]]] = [
        [  # x0 <= 2  -->  in standard form: x0 + (-2) <= 0
           ((linear_prog.ColKind.VAR, 0), linear_prog.Sign.POS, 1.0),
           ((linear_prog.ColKind.AFFINE, None), linear_prog.Sign.POS, -2.0)
        ]
    ]
    lp = LPmod.init(lambda j: f"x{j}", 1, objective = objective, ineqs = ineqs)
    basic_point = np.array([0.0])  # Clearly feasible point
    
print(f"Feasible point: {lp.is_point_feasible(basic_point)}")

# === 3. Tangent digraph ===
# === 3. Search for a basic point ===
print("Searching for a valid basic point...")
try:
    # First check if the point from file is already a basic point
    tg = tangent_digraph.TangentDigraph.compute(lp, basic_point)
    print("Tangent digraph computed successfully")
    
    if tg.is_basic_point():
        print("The point from file is already a basic point!")
    else:
        print("The point from file is not a basic point. Continuing anyway to test the algorithm...")
    
    tg.print()
    print("Is basic point:", tg.is_basic_point())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# === 4. Simplet ===
print("Initializing Simplet...")

try:
    Simp = Simplet(LPmod)
    print("Simplet created successfully")
    
    # Try to create the instance, handling the exception if the point is not basic
    try:
        instance = Simp.init(lp, basic_point)
        print("Instance created successfully")
    except ValueError as e:
        if "punto base" in str(e):
            print("The point is not a basic point. Terminating here for now.")
            print("To continue, we need to implement basic point search or modify the algorithm.")
            exit(0)
        else:
            raise
    
    # Check if the print method exists
    if hasattr(Simp, 'print_status'):
        Simp.print_status(instance, None)
    else:
        print("Instance status:")
        print(f"  Current point: {instance.point}")
        if hasattr(instance, 'reduced_costs'):
            print(f"  Number of reduced costs: {len(instance.reduced_costs)}")
    
    print("Starting resolution...")
    Simp.solve(instance, Simp.bland_rule)
    print("Resolution completed!")
    
except Exception as e:
    print(f"Error using Simplet: {e}")
    import traceback
    traceback.print_exc()
