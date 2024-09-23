import numpy as np
from scipy.linalg import lu, inv, hilbert, eigvals
from numpy.linalg import cond, det, solve

# Reading matrix A and vectors b1 and b2 from a file
#file_name = 'text.txt'
def read_matrix_and_vectors(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    A = data[:5, :5]  # 5x5 matrix
    b1 = data[5, :]   # first vector
    b2 = data[6, :]   # second vector
    return A, b1, b2
#does this change in vscode
# Perform power method to find largest eigenvalue
def power_method(A, num_iterations=1000, tol=1e-9):
    n, _ = A.shape
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)
    
    for _ in range(num_iterations):
        b_next = np.dot(A, b)
        b_next = b_next / np.linalg.norm(b_next)
        
        # Check for convergence
        if np.linalg.norm(b - b_next) < tol:
            break
        b = b_next
    
    eigenvalue = np.dot(b.T, np.dot(A, b)) / np.dot(b.T, b)
    return eigenvalue


# Function to generate the characteristic polynomial from eigenvalues
def eigenvalue_polynomial(eigenvalues):
    # Generate the coefficients of the polynomial whose roots are the eigenvalues
    polynomial_coefficients = np.poly(eigenvalues)  # np.poly generates the coefficients
    # Create a readable polynomial equation
    characteristic_polynomial = np.poly1d(polynomial_coefficients)
    
    return characteristic_polynomial

# Example usage


# Load matrix A and vectors b1, b2 from file
A, b1, b2 = read_matrix_and_vectors(r'C:\Users\hghit\Python\numerical_methods\matrix.csv')

# 1. LU Decomposition and Eigenvalue Calculation
P_matrix, L, U = lu(A)
eigenvalues_A = eigvals(A)

print("Matrix A:")
print(A)
print("\nVector b1:")
print(b1)
print("\nVector b2:")
print(b2)

print("\nEigenvalues of A using LU decomposition:")
print(eigenvalues_A)

# 2. Determinant and Uniqueness
det_A = np.prod(eigenvalues_A)
print("\nDeterminant of A:")
print(det_A)

if det_A != 0:
    print("The system has a unique solution.")
else:
    print("The system does not have a unique solution.")

# 3. Condition Number and Hilbert Comparison
cond_A = np.max(eigenvalues_A) / np.min(eigenvalues_A)
hilbert_5 = hilbert(5)
cond_hilbert = cond(hilbert_5)

print("\nCondition number of A:")
print(cond_A)
print("Condition number of the 5x5 Hilbert matrix:")
print(cond_hilbert)

if cond_A > cond_hilbert:
    print("A is more ill-conditioned than the Hilbert matrix.")
else:
    print("A is better conditioned than the Hilbert matrix.")

# 4. Polynomial Equation with Eigenvalues as Roots
#polynomial = np.poly(eigenvalues_A)  # Use numpy.poly to get the coefficients of the characteristic polynomial
#print(np.poly1d(polynomial))

polynomial = eigenvalue_polynomial(eigenvalues_A)
print("\nThe characteristic polynomial is:")
print(polynomial)

# 5. Power Method for Eigenvalue Calculation
largest_eigenvalue_A = power_method(A)
print("\nLargest eigenvalue of A using the power method:")
print(largest_eigenvalue_A)

# Power method for inverse of A
if(det(A) != 0):
    A_inv = inv(A)

    largest_eigenvalue_A_inv = power_method(A_inv)
    print("\nLargest eigenvalue of A inverse using the power method:")
    print(largest_eigenvalue_A_inv)
else:
    print("A is singular matirx!\n")

# 6. Solve Ax = b for two b vectors
if(det(A) != 0):
    x1 = solve(A, b1)
    print("\nSolution for Ax = b1:")
    print(x1)

    x2 = solve(A, b2)
    print("\nSolution for Ax = b2:")
    print(x2)
