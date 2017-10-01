import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt


def ell_valign(v,x):

    #check for dimension, raise error

    U1, s1, V1 = np.linalg.svd(v,full_matrices=False)
    U2, s2, V2 = np.linalg.svd(x,full_matrices=False)

    #print U1.shape, s1.shape, V1.shape

    T = np.linalg.multi_dot([U1, V1, V2.T, U2.T])

    return T



def main():
    v = np.random.randn(5,1)
    x = np.random.randn(5,1)

    #print v
    #print x
    S = ell_valign(v,x)

    print S
    #print U.shape, V.shape, s.shape
    #S = np.zeros((9, 6), dtype=complex)
    #S[:6, :6] = np.diag(s)
    #print np.allclose(a, np.dot(U, np.dot(S, V)))


if __name__ == "__main__":
    main()