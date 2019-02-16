import numpy as np
from scipy import linalg



class System(object):
    def __init__(self,t_end):
        # setup target system model here
        self.A = []
        self.F = []
        self.BPB = []
        self.L0 = []
        self.xc = []
        self.Bp = []

        self.num_search = 2
        self.n = 2
        self.abs_tol = 0.0001
        self.t0 = 0.000
        self.t1 = t_end

        self.xc = np.array([[0], [0]])
        self.Bp = np.array([[0], [0]])
        # A = np.array([[0, -10],
        #              [2, -8],
        #              ], dtype='Float64')
        # sys.A = np.reshape(A, (2, 2)) #packed in matrix form

        A = np.array([[0.000], [-10.00],
                      [2.0000], [-8.000],
                      ])
        self.A = np.reshape(A, (2, 2))  # packed in matrix form
        B = np.array([[10, 0],
                      [0, 2],
                      ])
        self.B = np.reshape(B, (2, 2))  # packed in matrix form
        P = np.array([[1, 0],
                      [0, 1],
                      ])
        self.P = np.reshape(P, (2, 2))  # packed in matrix form
        self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
        self.L = np.array([[1, 0],
                          [0, 1]
                          ])
        # X0 =  np.array([[1, 0],
        #              [0, 1]
        #              ], dtype='Float64')

        X0 = np.array([[1.000], [0.000],
                       [0.000], [1.000],
                       ])
        self.X0 = np.reshape(X0, (2, 2))  # packed in matrix form
        # sys.X0 = np.reshape(X0, (2, 2))  # packed in matrix form

        M = linalg.sqrtm(self.X0)

        # M = 0.5 * (M + M.T)
        # sys.M = np.reshape(0.5*(M+M.T),sys.n*sys.n,1)
        self.M = 0.5 * (M + M.T)
        print("M: " + str(self.M))
        # xl0 = M * l0

        # status check here, rank check


        self.time_grid_size = 200
        self.time_grid, self.dt = np.linspace(self.t0, self.t1, self.time_grid_size, dtype=float, retstep=True)
        print("sys dt " + str(self.dt))
        self.len_prop = self.time_grid_size

    def switch_system(self):
        # L2 = 1
        # R2 = 2
        # A2 = [0 -1/C; 1/L2 -R2/L2]
        # B2 = [1/C 0; 0 1/L2]
        A = np.array([[0, -10],
                      [1, -2],
                      ])

        self.A = np.reshape(A, (2, 2))  # packed in matrix form

        B = np.array([[10, 0],
                      [0, 1],
                      ])

        self.B = np.reshape(B, (2, 2))  # packed in matrix form

        P = np.array([[1, 0],
                      [0, 1],
                      ])

        self.P = np.reshape(P, (2, 2))  # packed in matrix form

        self.BPB = np.linalg.multi_dot([self.B, self.P, self.B.T])
