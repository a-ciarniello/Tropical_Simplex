import parser
import numeric
import linear_prog
import group
import numpy as np

def test_parser_with_lp_file(filename="problems/example_2.lp"):
    
    """Problem testing with an actual .lp file."""
    print(f"=== PROBLEM TEST: {filename} ===")
    
    with open(filename, 'r') as f:
        content = f.read()

    print("File content:")
    print(content)
    print()

    # Test 1: Header parsing test
    print("1. Test parsing header...")
    numeric_name, var_names = parser.lexer_header(content)
    print(f"   numeric_name: {numeric_name}")
    print(f"   var_names: {var_names}")
    print()

    # Test 2: Complete parsing test
    print("2. Complete parsing test...")
    Num = numeric.get(numeric_name)
    Parse = parser.Parser(Num)

    objective, ineqs, basic_point, is_maximize = Parse.main(content)
    print(f"   objective: {objective}")
    print(f"   \n ineqs: {ineqs}")
    print(f"   \n basic_point: {basic_point}")
    print(f"   \n is_maximize: {is_maximize}")
    print()

    # Test 3: LP creation test
    print("3. LP creation test...")
    G = group.GroupFromNumeric(Num)
    LPmod = linear_prog.LinearProg(G)
    
    nb_var = len(var_names)
    var_names_array = [""] * nb_var
    for name, idx in var_names.items():
        var_names_array[idx] = name
    var_names_fun = lambda j: var_names_array[j]
    
    lp = LPmod.init(var_names_fun, nb_var, objective, ineqs)
    print("   LP created successfully!")
    lp.pretty_print()
    print()

    # Test 4: Feasibility test
    print("4. Basic point feasibility test...")
    if basic_point.size > 0:
        feasible = lp.is_point_feasible(basic_point)
        print(f"   Basic point {basic_point} feasibility: {feasible}")

        # Test other points
        l = lp.dim()

        num_points = 3
        test_points = [np.round(np.random.uniform(0, 5, size=l), 1) for _ in range(num_points)]
    
        
        for point in test_points:
            feasible = lp.is_point_feasible(point)
            print(f"   Point {point}  feasibility: {feasible}")
    else:
        print("   No basic point provided.")

    print("The problem has been correctly parsed ===> Problem well defined.")
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:

        filename = sys.argv[1]
        test_parser_with_lp_file(filename)
    else:

        test_parser_with_lp_file()