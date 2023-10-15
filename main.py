# This program verifies some matrix identities for skew-symmetric matrices

import numpy as np
import sys
from tqdm import tqdm

def identity_check(n_dimension, n_examples, seed, tolerance):
    np.random.seed(seed)
    while True:
        try:
            n_dimension = int(n_dimension)
            if n_dimension < 0:
                sys.exit("Error: The dimension of the matrix X is not a positive integer.")
            else:
                break
        except ValueError as ex:
            sys.exit("Error: The dimension of the matrix X is not a positive integer.")
    while True:
        try:
            n_examples = int(n_examples)
            if n_examples < 0:
                sys.exit("Error: The number of examples is not a positive integer.")
            else:
                break
        except ValueError as ex:
            sys.exit("Error: The number of examples is not a positive integer.")
    while True:
        try:
            tolerance = float(tolerance)
            if tolerance < 0:
                sys.exit("Error: The tolerance is a negative number")
            else:
                break
        except ValueError as ex:
            sys.exit("Error: The tolerance is not a float.")
    print("Degree 3: X^3 = 1/2 tr(X^2)X")
    for i in tqdm(range(0,n_examples)):
        M = np.random.rand(n_dimension,n_dimension)
        X = (1/2) * (M - np.transpose(M))
        E = np.linalg.matrix_power(X, 3) - (1 / 2) * np.trace(np.linalg.matrix_power(X, 2)) * X
        error_L1 = np.sum(np.absolute(E))
        if error_L1 > tolerance:
            print("Degree 3 identity failed at dimension " + str(n_dimension) + " for the counterexample:")
            print(X)
            print("The absolute error is " + str(error_L1) + ".")
            break
        if i + 1 == n_examples:
            print("Verification of " + str(n_examples) + " examples in dimension " + str(n_dimension) + " complete.")

    print("Degree 5: X^5 - (1/2)tr(X^2)X^3 + (1/8)(tr(X^2)^2 - 2tr(X^4))X = 0")
    for i in tqdm(range(0, n_examples)):
        M = np.random.rand(n_dimension, n_dimension)
        X = (1 / 2) * (M - np.transpose(M))
        pf2 = (np.trace(np.linalg.matrix_power(X, 2)) ** 2 - 2*np.trace(np.linalg.matrix_power(X, 4)))/8
        E = np.linalg.matrix_power(X, 5) - (1/2)*np.trace(np.linalg.matrix_power(X, 2)) * np.linalg.matrix_power(X, 3) + pf2 * X
        error_L1 = np.sum(np.absolute(E))
        if error_L1 > tolerance:
            print("Degree 5 identity failed at dimension " + str(n_dimension) + " for the counterexample:")
            print(X)
            print("The absolute error is " + str(error_L1) + ".")
            break
        if i + 1 == n_examples:
            print("Verification of " + str(n_examples) + " examples in dimension " + str(n_dimension) + " complete.")

    return

identity_check(5,1000,1999,10**-14)





