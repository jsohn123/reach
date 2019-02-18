import numpy as np
from scipy import linalg

"""configures time grid, propagation, tolerance"""
#class EllOptions(object):
#    def __init__(self):
#        self.num_search = 2
#        self.n = 2
#        self.abs_tol = 0.0001
#        self.t0 = 0.000
#        self.t1 = t_end
#        self.time_grid_size = 200
#        self.time_grid, self.dt = np.linspace(self.t0, self.t1, self.time_grid_size, dtype=float, retstep=True)
#        self.len_prop = self.time_grid_size



"""contains dynamical system information to propagate"""
class EllSystem(object):
    def __init__(self,t_end=1):
        """setup target system model here"""

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

        self.num_search = 2
        self.n = 2
        self.abs_tol = 0.0001
        self.t0 = 0.000
        self.t1 = t_end
        self.time_grid_size = 200
        self.time_grid, self.dt = np.linspace(self.t0, self.t1, self.time_grid_size, dtype=float, retstep=True)
        self.len_prop = self.time_grid_size




    def switch_system(self):

        self.A = np.matrix('0.000 -10.000; 1.000 -2.000')

        self.B = np.matrix('10.000 0.000; 0.000 1.000')

        self.P = np.matrix('1.000 0.000; 0.000 1.000')

        self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
