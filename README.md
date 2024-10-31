# P2_Optimization_Operational_Research - LEFEBVRE Romain Group B

## Overview

This project aims to solve linear optimization problems using Google OR Tools and a step by step basic Simplex method. 

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)

## Features

- **OR-Tools Integration**
- **Step-by-Step Simplex Method**
- **Multiple Solutions Support**: Finds alternative optimal solutions when they exist (for more precision view the Known Issues section)
- **Handles Unbounded and Infeasible Cases**: Output when the feasible region is unbounded or no feasible solution exists.
- **Handles N-Dimensions Problem**

## Getting Started

### Prerequisites

- **Python 3.X**
- **Numpy**
- **Pandas**
- **Google OR Tools**

You can install them using pip:

```bash
pip install numpy pandas ortools
```
### Execution

The program accepts the number of variables, constraints, and the coefficients of the objective function and constraints as command-line arguments:

- num_vars: Number of variables in the problem.
- num_constraints: Number of constraints in the problem.
- c: Coefficients of the objective function.
- A: Coefficients of the constraints in row-major order.
- b: Right-hand side (RHS) values for each constraint.

Here is an example with the following problem:

maximize z = 3x1 + x2

subject to
- x2 ≤ 5
- x1 + x2 ≤ 10
- −x1 + x2 ≥ −2
- x1, x2 ≥ 0

```bash
python P2_LEFEBVRE_Romain_Group_B.py --c 3 1 --A 0 1 1 1 1 -1 --b 5 10 2 --num_vars 2 --num_constraints 3
```
You can get some help with the following command:
```bash
python P2_LEFEBVRE_Romain_Group_B.py -h
```
### Output

OR-Tools Solver: Prints the optimal solution and objective value.

Step by Step Simplex Method: Displays each tableau step-by-step, showing the pivot operations, basic and non-basic variables, and checks for alternative solutions.

## Known Issues

When dealing with a problem with multiple optimal solutions, the function using OR Tools to solve the problem will not return all the optimal solutions of the problem as the solver GLOP returns by default only one optimal solution.
For a 2D problem, it will return a solution which is not on the corners of the feasible region but rather on the corresponding side of the feasible region.
