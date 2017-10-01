import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode

from scipy import linalg
from scipy import special, optimize
import matplotlib.pyplot as plt

def ell_valign(v,x):

    #check for dimension, raise error

    U1, s1, V1 = np.linalg.svd(v,full_matrices=False)
    U2, s2, V2 = np.linalg.svd(x,full_matrices=False)

    #print U1.shape, s1.shape, V1.shape

    T = np.linalg.multi_dot([U1, V1, V2.T, U2.T])

    return T

def ee_ode_feed():
    print "this is where we calculate the dX/dt in shape matrix"

def deriv(t,A,Ab):
    A = np.reshape(A,(3,3))

    print "got it back"
    print A
    rate = np.dot(Ab, A)


    #print "rate"
    #print rate

    return np.reshape(rate,3*3) #vectorize


class System(object):
    def __init__(self):
        self.A = []
        self.F = []
        self.BPB = []
        self.L0=[]

def deriv_ea_nodist(t,y,n,sys):

    # "received" + str(y)
    X = np.reshape(y, (n, n)) #back to matrix
    A = sys.A
    F = sys.F
    BPB = sys.BPB

    L0 = sys.L0

    p1 = np.sqrt(np.linalg.multi_dot([L0.T, F, BPB, F.T, L0]))
    #print "p1" + str(p1)

    #print "F"+ str(F)
    #print "L0" + str(L0)
    #print "X" + str(X)
    #print "y" + str(y)
    p2 = np.sqrt(np.linalg.multi_dot([L0.T, F, X, F.T, L0]))
    #print "p2" + str(p2)


    pp1 = p1/p2

    #print "pp1" + str(pp1)

    pp2 = p2/p1
    #print "pp2" + str(pp2)
    dxdt = np.dot(A,X) + np.dot(X,A.T) + pp1*X + pp2*BPB

    #print "dxdt" + str(dxdt)


    dxdt = np.reshape(0.5*(dxdt+dxdt.T),n*n,1)

    #print "dxdt2" + str(dxdt)

    return dxdt

def update_sys(sys,t_now,t0):

    #sys = System()
    #A = np.array([[1, 0, 0],
    #              [0, 1, 0],
    #              [0, 0, 1]])
#
    #sys.A = np.reshape(A, (3, 3))

    sys.F = linalg.expm((sys.A)*(t_now-t0))

    #F = np.array([[1, 0, 0],
    #             [0, 1, 0],
    #             [0, 0, 1]])
    #sys.F = np.reshape(F, (3, 3))

    #BPB = np.array([[1, 0, 0],
    #              [0, 1, 0],
    #              [0, 0, 1]])
    #sys.BPB = np.reshape(BPB, (3, 3))

    #sys.L0 = np.array([[1, 0, 0]]).T

    return sys



def system_init():

    #setup target system model here
    sys = System()

    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])



    sys.A = np.reshape(A, (3, 3)) #packed in matrix form


    B = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    sys.B = np.reshape(B, (3, 3))  # packed in matrix form


    P =  np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    sys.P = np.reshape(P, (3, 3))  # packed in matrix form


    sys.BPB = np.linalg.multi_dot([sys.B, sys.P, sys.B ])

    sys.L = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    X0 =  np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    #X0= np.array([[np.random.random_integers(1,5), 0, 0],
    #               [0, np.random.random_integers(1,5), 0],
    #               [0, 0, np.random.random_integers(1,5)]])

    sys.X0 = np.reshape(X0, (3, 3))  # packed in matrix form

    sys.num_search = 3
    sys.n = 3

    sys.t0 = 0
    sys.t1 = 3
    sys.dt=0.01

    sys.len_prop = int(np.ceil(float((sys.t1 - sys.t0) / sys.dt))) + 2
    return sys


def reach_per_search(sys):
    #returns timeseries evolution of reachable set
    y0 = np.reshape(sys.X0, 3 * 3)

    t0 = sys.t0
    t1 = sys.t1
    dt = sys.dt

    len_prop = sys.len_prop
    tube = np.zeros((sys.n*sys.n, len_prop), dtype=np.float)

    r = ode(deriv_ea_nodist).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(sys.n)

    tube[:,0] = y0

    i = 1
    while r.successful() and r.t < t1:
        update_sys(sys, r.t, t0)  # update transition mat phi
        # update sys
        r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)
        tube[:,i] = r.y
        i += 1
        #print i




        # print("%g %g" % (r.t, r.y))
        # print r.t
        # print "result"
        # print np.reshape(r.y, (n, n))

    # print r.t
    #print "done"
    # print r.y
    #print np.reshape(r.y, (sys.n, sys.n))
    #print np.reshape(tube[:,-1], (sys.n, sys.n))

    return tube

def reach():
    sys = system_init()

    #center calculation here


    reach_set =np.empty((sys.n*sys.n, sys.len_prop, sys.num_search))
    #initialize the shape matrix tube

    for i in range(sys.num_search):
        sys.L0 = sys.L[i].T
        tube = reach_per_search(sys)
        reach_set[:,:,i] = tube


    #graphing here

    print np.reshape(reach_set[:,-1,0], (sys.n, sys.n))

def ode_integrate():
    #Ab = np.array([[-0.25, 0, 0],
    #               [0.25, -0.2, 0],
    #               [0, 0.2, -0.1]])

    Ab = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    time = np.linspace(0, 25, 101)
    #A0 = np.array([10, 20, 30])
    Amat = np.random.randn(3, 3)
    A0 = np.reshape(Amat, 3*3)

    print "Amat"
    print Amat
    print A0
    #rate = np.dot(Ab, A0)

    #print "rate"
    #print rate

    t0=0
    r = ode(deriv).set_integrator('dopri5')
    r.set_initial_value(A0, t0).set_f_params(Ab)
    t1 = 10
    dt = 0.01
    print "start"
    print A0

    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        #print("%g %g" % (r.t, r.y))
        #print r.t
        #print "result"
        #print r.y

    print r.t
    print "done"
    print r.y


def main():
    #v = np.random.randn(5,1)
    #x = np.random.randn(5,1)
    #S = ell_valign(v,x)
    #print S
    print "ODE integration example"

    reach()
    #MA= ode_example()

    #print MA
    #print MA.shape
    #print U.shape, V.shape, s.shape
    #S = np.zeros((9, 6), dtype=complex)
    #S[:6, :6] = np.diag(s)
    #print np.allclose(a, np.dot(U, np.dot(S, V)))


if __name__ == "__main__":
    main()