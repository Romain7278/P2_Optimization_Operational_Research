import argparse
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

def parse_arguments():
    """
    Parse command-line arguments for the linear programming problem.
    
    Returns:
        Parsed arguments containing the problem parameters.
    """
    parser = argparse.ArgumentParser(description="Solve a linear programming (LP) problem using the Simplex method with OR-Tools and a step-by-step approach.")
    parser.add_argument('--num_vars', type=int, required=True, help="Number of variables in the problem.")
    parser.add_argument('--num_constraints', type=int, required=True, help="Number of constraints in the problem.")
    parser.add_argument('--c', nargs='+', type=float, required=True, help="Objective function coefficients (1 for each variable).")
    parser.add_argument('--A', nargs='+', type=float, required=True, help="Constraint coefficients in row order.")
    parser.add_argument('--b', nargs='+', type=float, required=True, help="Constraint right-hand side (RHS) values (1 for each constraint).")

    return parser.parse_args()

def Solving_OR(c, A, b):
    # Initialize a GLOP solver instance
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # Define variables and store them in a list
    vars = []
    for i in range(len(c)):
        vars.append(solver.NumVar(0, solver.infinity(), f'x_{i}'))

    print("Number of variables =", solver.NumVariables())

    # Define constraints and assign coefficients for each variable
    for i in range(len(A)):
        constraint = solver.RowConstraint(0, b[i], f'c{i}')
        for j in range(len(c)):
            constraint.SetCoefficient(vars[j], A[i][j])

    print("Number of constraints =", solver.NumConstraints())

    # Define the objective function and set it for maximization
    objective = solver.Objective()
    for j in range(len(c)):
        objective.SetCoefficient(vars[j], c[j])
    objective.SetMaximization()

    # Solve the problem and output the solution if optimal
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        solution = [vars[j].solution_value() for j in range(len(c))]
        print("\nSolution:")
        print("Optimal Objective Value:", objective.Value())
        print("Optimal Solution:", solution)
    elif status == pywraplp.Solver.UNBOUNDED:
        print("The feasible region is unbounded, so an optimal solution cannot be found.")
    else:
        print("The problem does not have an optimal solution in the feasible region.")
    return None, None

def Step_by_step_Simplex(c, A, b):
    num_vars = len(c)
    num_constraints = len(b)

    # Construct the initial tableau for the Simplex method
    table = np.zeros((num_constraints + 1, num_vars + num_constraints + 2))
    table[:-1, :num_vars] = A  # Coefficients of constraints
    table[:-1, num_vars:num_vars + num_constraints] = np.eye(num_constraints)  # Identity matrix for slack variables
    table[:-1, -2] = 0  # Objective function column for constraints
    table[:-1, -1] = b  # RHS values
    table[-1, :num_vars] = -np.array(c)  # Objective function coefficients (in negative for maximization)
    table[-1, -2] = 1  # Coefficient for the objective variable in objective row

    # Create names for columns and rows in the tableau
    column_names = [f"x{i+1}" for i in range(num_vars)] + \
                   [f"s{i+1}" for i in range(num_constraints)] + ["z", "RHS"]
    row_names = [f"s{i+1}" for i in range(num_constraints)] + ["z"]

    # Initialize lists for basic and non-basic variables
    basic_vars = list(range(num_vars, num_vars + num_constraints))
    non_basic_vars = list(range(num_vars))

    print(f"\nInitial table:")
    table_df = pd.DataFrame(table, index=row_names, columns=column_names)
    print(table_df)

    step = 1
    while True:
        # Check if current solution is optimal
        if all(table[-1, :-2] >= 0):
            
            solution = np.zeros(num_vars)
            for i, var in enumerate(basic_vars):
                if var < num_vars:
                    solution[var] = table[i, -1]
            objective_value = table[-1, -1]

            # Verify solution feasibility
            if any(solution < 0):
                print("The solution is not feasible.")
                return None, None
            
            print("\nOptimal solution found.")
            print("Optimal Objective Value:", objective_value)
            print(f"Optimal Solution: {solution}")
            
            # Check for alternative optimal solutions by identifying zero reduced costs
            alternative_solutions = []
            for i, var in enumerate(non_basic_vars):
                if table[-1, var] == 0:
                    # Explore an alternative solution by pivoting on this non-basic variable
                    alt_table = table.copy()
                    entering = var
                    ratios = [alt_table[i, -1] / alt_table[i, entering] if alt_table[i, entering] > 0 else np.inf for i in range(num_constraints)]
                    leaving = np.argmin(ratios)
                    pivot_value = alt_table[leaving, entering]

                    row_names = [column_names[entering] if x == row_names[leaving] else x for x in row_names]
                    
                    # Perform the pivot operation for the alternative solution
                    alt_table[leaving, :] /= pivot_value
                    for j in range(num_constraints + 1):
                        if j != leaving:
                            alt_table[j, :] -= alt_table[j, entering] * alt_table[leaving, :]

                    print(f"\nStep {step}:")
                    table_df = pd.DataFrame(alt_table, index=row_names, columns=column_names)
                    print("Current table:")
                    print(table_df)
                    step += 1
                    
                    # Record the alternative solution
                    alt_solution = np.zeros(num_vars)
                    for k, basic in enumerate(basic_vars):
                        if basic < num_vars:
                            alt_solution[basic] = alt_table[k, -1]
                    alternative_solutions.append((alt_solution, alt_table[-1, -1]))
                    print("\nAlternative optimal solution found:")
                    print(f"Objective Value: {alt_table[-1, -1]}")
                    print(f"Alternative Solution: {alternative_solutions[-1][0]}")
                    
            return solution, objective_value
        
        # Find the entering variable with the most negative coefficient in the objective row
        entering = np.argmin(table[-1, :-2])

        # Check if the feasible region is unbounded
        if all(table[:-1, entering] <= 0):
            print("The feasible region is unbounded.")
            return None, None

        print(f"\nEntering basic variable: {column_names[entering]}")
        
        # Determine leaving variable by the minimum ratio rule
        ratios = []
        for i in range(num_constraints):
            if table[i, entering] > 0:
                ratios.append(table[i, -1] / table[i, entering])
            else:
                ratios.append(np.inf)
        leaving = np.argmin(ratios)
        pivot_value = table[leaving, entering]
        print(f"Leaving basic variable: {row_names[leaving]}")
        print(f"Pivot value: {pivot_value}")

        row_names = [column_names[entering] if x == row_names[leaving] else x for x in row_names]

        # Perform pivot operation to update tableau
        table[leaving, :] /= pivot_value
        for i in range(num_constraints + 1):
            if i != leaving:
                table[i, :] -= table[i, entering] * table[leaving, :]
        
        # Update basic and non-basic variables
        basic_vars[leaving], non_basic_vars[entering] = non_basic_vars[entering], basic_vars[leaving]
        print(f"\nStep {step}:")
        table_df = pd.DataFrame(table, index=row_names, columns=column_names)
        print("Current table:")
        print(table_df)
        step += 1

if __name__ == "__main__":
    # Parse command-line arguments to define the problem
    args = parse_arguments()
    # Reshape input coefficients for constraints
    A = np.array(args.A).reshape(args.num_constraints, args.num_vars)
    b = np.array(args.b)
    c = np.array(args.c)

    print("\n--- Using OR-Tools ---")
    Solving_OR(c, A, b)

    print("\n--- Using Simplex Method ---")
    Step_by_step_Simplex(c, A, b)