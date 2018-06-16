import numpy as np
from scipy.integrate import ode
from scipy import linalg
import sys
import ellipsoidal_visualizer as ev

class System(object):
    def __init__(self):
        self.A = []
        self.F = []
        self.BPB = []
        self.L0=[]
        self.xc =[]
        self.Bp = []

def switch_system(sys):
    #L2 = 1
    #R2 = 2
    #A2 = [0 -1/C; 1/L2 -R2/L2]
    #B2 = [1/C 0; 0 1/L2]
    A = np.array([[0, -10],
                  [1, -2],
                  ])

    sys.A = np.reshape(A, (2, 2))  # packed in matrix form

    B = np.array([[10, 0],
                  [0, 1],
                  ])

    sys.B = np.reshape(B, (2, 2))  # packed in matrix form

    P = np.array([[1, 0],
                  [0, 1],
                  ])

    sys.P = np.reshape(P, (2, 2))  # packed in matrix form

    sys.BPB = np.linalg.multi_dot([sys.B, sys.P, sys.B.T])

def system_init_LC():

    #setup target system model here
    sys = System()
    sys.xc = np.array([[0.5],[0.5]])
    sys.Bp = np.array([[0],[0]])
    A = np.array([[0, -10],
                  [2, -8],
                  ])
    sys.A = np.reshape(A, (2, 2)) #packed in matrix form
    B = np.array([[10, 0],
                  [0, 2],
                  ])
    sys.B = np.reshape(B, (2, 2))  # packed in matrix form
    P =  np.array([[1, 0],
                  [0, 1],
                  ])
    sys.P = np.reshape(P, (2, 2))  # packed in matrix form
    sys.BPB = np.linalg.multi_dot([sys.B, sys.P, sys.B.T ])
    sys.L = np.array([[1, 0],
                      [0, 1]
                      ])
    X0 =  np.array([[1, 0],
                  [0, 1]
                  ])
    sys.X0 = np.reshape(X0, (2, 2))  # packed in matrix form

    #status check here, rank check
    sys.num_search = 2
    sys.n = 2
    sys.abs_tol = 0.0001
    sys.t0 = 0.0
    sys.t1 = 5.00

    sys.time_grid_size = 200
    sys.time_grid, sys.dt= np.linspace(sys.t0,sys.t1,sys.time_grid_size,dtype=float,retstep=True)
    print "sys dt " + str(sys.dt)
    sys.len_prop = sys.time_grid_size
    return sys

def reach_center(sys,evolve = False, y0=[], debug =False):
    #center calculation here

    if evolve:
        print "evolving center from latest result"
        #print y0
    else:
        y0 = np.reshape(sys.xc, (sys.n))
    t0 = float(sys.t0)
    t1 = float(sys.t1)
    dt = float(sys.dt)

    len_prop = sys.len_prop
    center_trajectory = np.zeros((sys.n, len_prop+1), dtype=np.float) #check this 04/03

    r = ode(deriv_reach_center).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(sys)

    center_trajectory[:, 0] = y0

    i = 1
    if debug:
        print "starting from" + str(r.y)
        print "buffer length: " + str(len_prop)
        print "t0: " + str(t0)
        print "t1: " + str(t1)
    while r.successful() and (r.t <= t1): #watch out for comparison error......

        #update_sys(sys, r.t, t0)  # update transition mat phi
        # update sys
        #r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)
        if debug:
            print "going for number: " +str(i)
        if float(r.t) > float(t1)+sys.abs_tol:
        #    print " r.t: " + str(r.t) + " t1: " + str(t1)
            if debug:
                print "should exit now"
            return center_trajectory
        center_trajectory[:,i] = r.y
        if debug:
            print "center prop_num: " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1)
        i += 1

    return center_trajectory
def deriv_reach_center(t,y,sys):
    x = y #in vector form
    #print x
    A = sys.A
    #print A
    Bp = sys.Bp

    #Gq = sys.Gq

    #print np.dot(A, x)
    dxdt = np.dot(A, x) + Bp.T
    #dxdt = A * x + Bp
    #print "dxdt" +str(dxdt)
    return np.reshape(dxdt,(sys.n,1))

def reach_per_search(sys, evolve = False, y0 = [],debug = False):
    #returns timeseries evolution of reachable set per direction of search vector from L0

    if evolve:
        print "evolving from latest tube result"
    else:
        y0 = np.reshape(sys.X0, sys.n * sys.n) #3*3

    t0 = sys.t0
    t1 = sys.t1
    dt = sys.dt

    len_prop = sys.len_prop
    if debug:
        print "len_prop: " + str(len_prop)
    tube = np.zeros((sys.n*sys.n,len_prop), dtype=np.float)
    time_tube = np.zeros(len_prop,dtype=np.float)

    r = ode(deriv_ea_nodist).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(sys.n)

    tube[:,0] = y0
    time_tube[0] = t0

    if debug:
        print "Y0, T0"
        print y0
        print t0
        print "END"
    i = 1
    while r.successful() and (r.t <= t1):
        #print "starting loop with: " + str(r.t)
        #update_sys(sys, r.t, t0)  # update transition mat phi
        sys.F = linalg.expm((sys.A) * (r.t - t0))
        # update sys
        r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)

        try:
            #append tube
            tube[:,i] = r.y
            time_tube[i] = r.t
        except Exception as e:
            if debug:
                print e
                print "tube[:,i] exited at " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1) + " dt: " + str(dt)
            return tube, time_tube
        #if float(r.t) >= float(t1)+sys.abs_tol:
        #    if debug:
        #        print " r.t: " + str(r.t) + " t1: " + str(t1)
        #        print "should exit now"
        #        try:
        #            print tube[:,i]
        #        except:
        #            print "nothing!"
        #    return tube,time_tube
        if debug:
            print "tube_prop_number: " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1)
        i += 1
    if debug:
        print "finished at " + str(r.t)
    return tube, time_tube
def deriv_ea_nodist(t,y,n,sys):

    # "received" + str(y)
    X = np.reshape(y, (n, n)) #back to matrix
    A = sys.A
    #F = sys.F #gives complete bogus
    F = np.linalg.inv(sys.F) #verify THIS PROBABLY SINCE ADJOINT EQUATION IS -A
    BPB = sys.BPB

    L0 = sys.L0

    p1 = np.sqrt(np.linalg.multi_dot([L0.T, F, BPB, F.T, L0]))

    p2 = np.sqrt(np.linalg.multi_dot([L0.T, F, X, F.T, L0]))
    if abs(p1) < sys.abs_tol:
        p1 = sys.abs_tol
    if abs(p2) < sys.abs_tol:
        p2 = sys.abs_tol


    pp1 = p1/p2

    pp2 = p2/p1

    dxdt = np.dot(A,X) + np.dot(X,A.T) + pp1*X + pp2*BPB

    dxdt = np.reshape(0.5*(dxdt+dxdt.T),n*n,1)


    return dxdt

def evolve_nodist(prev_reach_set,time_tube,prev_center_trajectory,sys, extra_time,debug = False):
    switch_system(sys) #only update system matrix, not

    sys.t0 = sys.t1 #continuous. hence, 1 time stamp overlap
    sys.t1 = sys.t1 + extra_time
    sys.time_grid, sys.dt = np.linspace(sys.t0, sys.t1, sys.time_grid_size, dtype=float, retstep=True)
    extra_len_prop = sys.time_grid_size

    prev_len_prop = sys.len_prop
    sys.len_prop = extra_len_prop

    if debug:
        print "prev_len_prop" + str(prev_len_prop)
        print "extra_len_prop" + str(extra_len_prop)

    evolved_center_trajectory = reach_center(sys,evolve = True, y0 = prev_center_trajectory[:,-1]) #resume from last element

    reach_set = np.empty((sys.n * sys.n, sys.len_prop, sys.num_search))

    for i in range(sys.num_search):
        sys.L0 = sys.L[i].T
        tube, extra_time_tube = reach_per_search(sys, evolve= True, y0 = prev_reach_set[:,-1,i]) #resume from last element
        reach_set[:,:,i] = tube    #time_series x vectorized shape matrix x search direction
        if debug:
            print "tube shape: " + str(tube.shape)
            print reach_set.shape
            print "last of tube: "
            print tube[:,-1]

    #NOW STITCH THE TUBES AND CENTER TRAJECTORY, MINDFUL OF 1 SAMPLE OVERLAP
    combined_reach_set = np.concatenate((prev_reach_set[:,:-1,:],reach_set),axis = 1)
    combined_center_trajectory = np.concatenate((prev_center_trajectory[:,:-1],evolved_center_trajectory),axis = 1)

    sys.len_prop = prev_len_prop + extra_len_prop-1

    combined_time_tube = np.append(time_tube[:prev_len_prop-1],extra_time_tube)

    return combined_reach_set, combined_center_trajectory, combined_time_tube

def inspect_slice(reach_set,sys):
    #expand this later to properly time stamped version
    print " AT TIME N=0"
    for i in range(sys.num_search):
        print "reach_set at " + str(i)+ "th direction"
        print sys.L[i].T
        print reach_set[:,0,i]

    print " AT TIME N=170"
    for i in range(sys.num_search):
        print "reach_set at " + str(i)+ "th direction"
        print sys.L[i].T
        print reach_set[:,170,i]
    print " AT TIME N=171"
    for i in range(sys.num_search):
        print "reach_set at " + str(i) + "th direction"
        print sys.L[i].T
        print reach_set[:, 171, i]

def reach():
    debug = False
    sys = system_init_LC() #SETUP DYNAMIC SYSTEM DESCRIPTION HERE

    print "starting center traj calculation"
    center_trajectory = reach_center(sys)
    print "done center traj calculation"

    reach_set =np.empty((sys.n*sys.n,sys.len_prop, sys.num_search))
    time_tube=[] #TIME_STAMP RECORD TO GUARANTEE CORRECT TIME SCALING OF THE GUI

    for i in range(sys.num_search):
        sys.L0 = sys.L[i].T
        tube,time_tube = reach_per_search(sys)
        reach_set[:,:,i] = tube    #time_series x vectorized shape matrix x search direction
        if debug:
            print "tube shape: " + str(tube.shape)
            print reach_set.shape
            print "last of tube: "
            print tube[:, -1]
    #inspect_slice(reach_set,sys)

    print "EVOLVE 1"
    evolved_reach_set, evolved_center_trajectory, evolved_time_tube = evolve_nodist(reach_set,time_tube,center_trajectory,sys,extra_time=4)

    print "EVOLVE 2"
    evolved_reach_set, evolved_center_trajectory,evolved_time_tube = evolve_nodist(evolved_reach_set,evolved_time_tube, evolved_center_trajectory, sys, extra_time=3)
#
    ev.reach_gui(sys,evolved_center_trajectory,evolved_reach_set[:,:,:], render_length = sys.len_prop,time_tube = evolved_time_tube)
def main():
    #v = np.random.randn(5,1)
    #x = np.random.randn(5,1)
    #S = ell_valign(v,x)
    #print S

    reach()
    #MA= ode_example()

    #print MA
    #print MA.shape
    #print U.shape, V.shape, s.shape
    #S = np.zeros((9, 6), dtype=complex)
    #S[:6, :6] = np.diag(s)
    #print np.allclose(a, np.dot(U, np.dot(S, V)))


if __name__ == "__main__":

    try:
        print sys.argv[1]
    except:
        print " no sys arg "
    main()