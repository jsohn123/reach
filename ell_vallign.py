import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt


def ell_valign(v,x):

    #check for dimension, raise error

    U1, s1, V1 = np.linalg.svd(v,full_matrices=True)
    U2, s2, V2 = np.linalg.svd(x,full_matrices=True)
    #has to be full SVD

    #sketch way of dealing with scalars
    if U1.shape[0] == 1 and U1.shape[1] ==1:
        U1 = np.asscalar(U1)
    if U2.shape[0] == 1 and U2.shape[1] == 1:
        U2_T = np.asscalar(U2)
    else:
        U2_T = U2.T
    if V1.shape[0]== 1 and V1.shape[1] == 1:
        V1 = np.asscalar(V1)
    if V2.shape[0] == 1 and V2.shape[1] ==1:
        V2_T = np.asscalar(V2)
    else:
        V2_T = V2.T
#
    T = U1*V1*V2_T*U2_T

    #T = U1*V1*V2.T*U2.T

    #T = np.linalg.multi_dot([U1, V1, V2.T, U2.T])

    return T



def main():
    #v = np.random.randn(2,1)
    #x = np.random.randn(2,1)
    xl0 = np.matrix('1; 0')
    l = np.matrix('2.4016; 9.3249')

    print xl0
    print l
    S = ell_valign(xl0,l)

    print S
    #print U.shape, V.shape, s.shape
    #S = np.zeros((9, 6), dtype=complex)
    #S[:6, :6] = np.diag(s)
    #print np.allclose(a, np.dot(U, np.dot(S, V)))


if __name__ == "__main__":
    main()