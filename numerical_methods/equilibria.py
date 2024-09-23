
import numpy as np
from scipy.linalg import lu, inv, hilbert, eigvals
from numpy.linalg import cond, det, solve, qr



def read_matrix_and_vectors(file_name):
    """Reading matrix A and vectors b1 and b2 from a file"""

    data = np.loadtxt(file_name, delimiter=',')

    A = data[:5, :5]  # 5x5 matrix
    b1 = data[5, :]   # first vector
    b2 = data[6, :]   # second vector

    return A, b1, b2




def check_difference_with_identity(L, U, tolerance=1e-10):
    """To check for the exit condition in the LU Decomposition to find the eigenvalues."""
    n = L.shape[0]

    # Check L and U with I
    for i in range(n):
        for j in range(n):
            # Check diagonal elements of L and U (should be close to 1)
            if i == j:
                if abs(L[i, j] - 1) > tolerance or abs(U[i, j] - 1) > tolerance:
                    return "Failure"
            # Check off-diagonal elements (should be close to 0)
            else:
                if abs(L[i, j]) > tolerance or abs(U[i, j]) > tolerance:
                    return "Failure"
    
    return "Success"




def eigenvalues_via_lu(A, tolerance=1, max_iterations=10000):
    """ Finds out the eigenvalues using LU decomposition as per  the method given by Sir"""

    n = A.shape[0]
    iterations = 0
    diff = np.inf

    while diff > tolerance and iterations < max_iterations:
        # LU decomposition using SciPy's built-in method
        P, L, U = lu(A)
        
        # Multiply U and L to get the next iteration of A
        A_next = U @ L
        
        # Check for convergence: comparing diagonals of A with A_next
        # diff = np.max(np.abs(np.diag(A_next) - np.diag(A)))

        flag = check_difference_with_identity(L, U, tolerance)
        if flag == 'Success':
            return np.diag(L), np.diag(U), iterations

        # Update A for the next iteration
        A = A_next
        iterations += 1

    # Return the diagonal as the eigenvalues and number of iterations
    #eigenvalues = np.diag(A)
    #return eigenvalues, iterations
    return "Could not get the eigenvalues."





def eigenvalues_via_qr(A, tolerance=1e-10, max_iterations=1000):
    """Finds eigenvalues using the QR decomposition method,
        returns the eigenvalues and the number of iteration that were needed."""
    
    n = A.shape[0]
    iterations = 0
    diff = np.inf

    while diff > tolerance and iterations < max_iterations:
        # QR decomposition
        Q, R = qr(A)
        
        # Multiply R and Q to get the next iteration of A
        A_next = R @ Q
        
        # Check for convergence by comparing the diagonals
        diff = np.max(np.abs(np.diag(A_next) - np.diag(A)))
        
        # Update A for the next iteration
        A = A_next
        iterations += 1

    # Return the diagonal as the eigenvalues and number of iterations
    eigenvalues = np.diag(A)
    return eigenvalues, iterations





def power_method(A, num_iterations=1000, tol=1e-9):
    """Finds the largest eigenvalues using the power method."""

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





def eigenvalue_polynomial(eigenvalues):
    """Generates a polynimial with given set of eigenvalues
       that is the characteristic equation."""
    
    # Generate the coefficients of the polynomial whose roots are the eigenvalues
    polynomial_coefficients = np.poly(eigenvalues)  # np.poly generates the coefficients
    # Create a readable polynomial equation
    characteristic_polynomial = np.poly1d(polynomial_coefficients)
    
    return characteristic_polynomial





def solve_using_lu(A, b):
    """Solves the given matrix equation using LU decomposition
    forward and backward substitution."""

    # Step 1: Perform LU decomposition
    P,L, U = lu(A)
    
    # Step 2: Forward substitution to solve Ly = b
    y = np.zeros_like(b)
    
    for i in range(len(y)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Step 3: Back substitution to solve Ux = y
    x = np.zeros_like(y)
    
    for i in range(len(x) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x





def print_section(title):
    """Define a simple function to print headers and dividers"""

    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")





def print_subsection(title):
    """Define a simple function to print sections and dividers"""

    print(f"\n{'-'*60}")
    print(f"{title:^60}")
    print(f"{'-'*60}")