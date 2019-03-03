import numpy as np
from scipy import linalg
import pdb

"""configures time grid, propagation, tolerance"""
def _is_squared(matrix):
    (n_row, n_col) = matrix.shape
    return n_row == n_col

"""contains dynamical system information to propagate, similart to linsys function of MATLAB"""
class EllSystem(object):
    def __init__(self,system_description = None,t_end=1):
        """setup target system model here"""

        if system_description is None:
            print("reverting to example system")
            self.A = np.matrix('0.000 -10.000; 2.000 -8.000')
            self.B = np.matrix('10.000 0.000; 0.000 2.000')
            self.P = np.matrix('1.000 0.000; 0.000 1.000')
            self.L = np.matrix('1.000 0.000; 0.000 1.000')
            self.X0 = np.matrix('1.000 0.000; 0.000 1.000')
            self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
            M = linalg.sqrtm(self.X0)
            self.M = 0.5 * (M + M.T)
            self.xc = np.matrix('0.000; 0.000')
            self.Bp = np.matrix('0.000; 0.000')

        else:
            try:
                self.A = system_description.get('A')
                self.B = system_description.get('B')
                self.P = system_description.get('P')
                self.L = system_description.get('L')
                self.X0 = system_description.get('X0')
                self.xc = system_description.get('XC')
                self.Bp = system_description.get('BC')
            except Exception as e:
                print(e)
                raise ValueError("Unable to import system_description. Check your matrix input")

            try:
                self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
                M = linalg.sqrtm(self.X0)
                self.M = 0.5 * (M + M.T)
            except Exception as e:
                print(e)
                raise AttributeError("BPB and M derivation failed")

        #system matrix A and ellipsoidal shape matrix should be square form
        if not (_is_squared(self.A) and _is_squared(self.X0) and _is_squared(self.P)):
            raise ValueError("The matrices A,P,X0 must be squared")

        # num_search and ndim derive from matrix shape after checks
        self.num_search = self.L.shape[1]
        self.n = self.A.shape[1]

        self.abs_tol = 0.0001
        self.t0 = 0.000
        self.t1 = t_end
        self.time_grid_size = 200
        self.time_grid, self.dt = np.linspace(self.t0, self.t1, self.time_grid_size, dtype=float, retstep=True)
        self.len_prop = self.time_grid_size

    def switch_system(self,system_description = None):

        if system_description is None:
            print("reverting to example system switch")
            self.A = np.matrix('0.000 -10.000; 1.000 -2.000')

            self.B = np.matrix('10.000 0.000; 0.000 1.000')

            self.P = np.matrix('1.000 0.000; 0.000 1.000')

            self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])

        else:
            try:
                self.A = system_description.get('A')
                self.B = system_description.get('B')
                self.P = system_description.get('P')

                # in system switch, only  A,B,P are updated. the rest cannot switch
            except Exception as e:
                print(e)
                ValueError("Unable to import system_description. Check your matrix input")

            try:
                self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
            except Exception as e:
                print(e)
                AttributeError("BPB derivation failed")


        #system matrix A and ellipsoidal shape matrix should be square form
        if not (_is_squared(self.A) and _is_squared(self.P)):
            raise ValueError("The matrices A,P must be squared")

        # num_search and ndim derive from matrix shape after checks
        if self.n != self.A.shape[1]:
            raise ValueError("The new matrix A is of different dimension!")
