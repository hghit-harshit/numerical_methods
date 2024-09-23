
from scipy.linalg import lu, inv, hilbert, eigvals
from numpy.linalg import cond, det, solve, qr

import numpy as np
from equilibria import*

# equilibria : custom library that contains all the functions that we wrote for the assignment 


# Load matrix A and vectors b1, b2 from file
A, b1, b2 = read_matrix_and_vectors(r'C:/Users/Atharva/code/python work/numerical methods/assignment.py/matrix.csv')




# 1. LU Decomposition and Eigenvalue Calculation

P_matrix, L, U = lu(A)
eigenvalues_A, iter = eigenvalues_via_qr(A)

# Output Matrix and Vectors
print_section("Matrix A and Vectors b1, b2")
print("Matrix A:")
print(np.array_str(A, precision=3, suppress_small=True))  # Limiting the precision for cleaner display
print("\nVector b1:")
print(np.array_str(b1, precision=3, suppress_small=True))
print("\nVector b2:")
print(np.array_str(b2, precision=3, suppress_small=True))

# Output Eigenvalues
print_section("1. Eigenvalue Calculation Using LU Decomposition")
print(f"{'Eigenvalue':<15}{'Value':>15}")
print(f"{'-'*30}")
for i, eig in enumerate(eigenvalues_A, start=1):
    print(f"Eigenvalue {i:<5}: {eig:>15.6f}")
print(f"\nNumber of iterations for QR decomposition: {iter}")

# Add spacing between sections
print("\n" * 2)




# 2. Determinant and Uniqueness

det_A = np.prod(eigenvalues_A)
print_section("2. Determinant of A and Uniqueness of Solution")
print(f"Determinant of A: {det_A:.6f}\n")

if det_A != 0:
    print("Result: The system has a unique solution.")
else:
    print("Result: The system does not have a unique solution.")

# Add spacing between sections
print("\n" * 2)




# 3. Condition Number and Hilbert Comparison

cond_A = np.max(eigenvalues_A) / np.min(eigenvalues_A)
hilbert_5 = hilbert(5)
cond_hilbert = cond(hilbert_5)

print_section("3. Condition Number Comparison")
print(f"{'Condition':<30}{'Value':>15}")
print(f"{'-'*45}")
print(f"Condition number of A: {cond_A:>15.6f}")
print(f"Condition number of the 5x5 Hilbert matrix: {cond_hilbert:>15.6f}")

if cond_A > cond_hilbert:
    print("\nResult: A is more ill-conditioned than the Hilbert matrix.")
else:
    print("\nResult: A is better conditioned than the Hilbert matrix.")

# Add spacing between sections
print("\n" * 2)




# 4. Polynomial Equation with Eigenvalues as Roots

polynomial = eigenvalue_polynomial(eigenvalues_A)
print_section("4. Characteristic Polynomial")
print("The characteristic polynomial is:\n")
print(np.poly1d(polynomial))

# Add spacing between sections
print("\n" * 2)




# 5. Power Method for Eigenvalue Calculation

largest_eigenvalue_A = power_method(A)
print_section("5. Largest Eigenvalue Using Power Method")
print(f"Largest eigenvalue of A using the power method: {largest_eigenvalue_A:.6f}")

# Add spacing between sections
print("\n" * 2)

# Power method for inverse of A
print_subsection("5.1 Power Method for A Inverse (if A is nonsingular)")
if det_A != 0:
    A_inv = inv(A)
    largest_eigenvalue_A_inv = power_method(A_inv)
    print(f"Largest eigenvalue of A inverse using the power method: {largest_eigenvalue_A_inv:.6f}")
else:
    print("A is a singular matrix, cannot calculate the inverse power method.")

# Add spacing between sections
print("\n" * 2)





# 6. Solve Ax = b for two b vectors

print_section("6. Solving Ax = b for Two b Vectors")
if det_A != 0:
    x1 = solve_using_lu(A, b1)
    x2 = solve_using_lu(A, b2)
    
    print(f"{'Solution':<15}{'Vector':>15}")
    print(f"{'-'*30}")

    print("Solution for Ax = b1:")
    print("\nVector b1:")
    print(np.array_str(b1, precision=3, suppress_small=True))
    print(np.array_str(x1, precision=6, suppress_small=True))

    print("\nSolution for Ax = b2:")
    print("\nVector b2:")
    print(np.array_str(b2, precision=3, suppress_small=True))
    print(np.array_str(x2, precision=6, suppress_small=True))

else:
    print("Cannot solve Ax = b as A is a singular matrix.")


print("\n\n\n\n\n\n\n")