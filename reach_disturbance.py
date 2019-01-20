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

def system_init(t_end):


    #setup target system model here
    sys = System()

    sys.num_search = 2
    sys.n = 2
    sys.abs_tol = 0.0001
    sys.t0 = 0.000
    sys.t1 = t_end

    sys.xc = np.array([[0],[0]], dtype = 'Float64')
    sys.Bp = np.array([[0],[0]], dtype='Float64')
    #A = np.array([[0, -10],
    #              [2, -8],
    #              ], dtype='Float64')
    #sys.A = np.reshape(A, (2, 2)) #packed in matrix form

    A = np.array([[0.000], [-10.00],
                  [2.0000], [-8.000],
                  ], dtype='Float64')
    sys.A = np.reshape(A, (2, 2))  # packed in matrix form
    B = np.array([[10, 0],
                  [0, 2],
                  ], dtype='Float64')
    sys.B = np.reshape(B, (2, 2))  # packed in matrix form
    P =  np.array([[1, 0],
                  [0, 1],
                  ], dtype='Float64')
    sys.P = np.reshape(P, (2, 2))  # packed in matrix form
    sys.BPB = np.linalg.multi_dot([sys.B, sys.P, sys.B.T ])
    sys.L = np.array([[1, 0],
                      [0, 1]
                      ], dtype='Float64')
    #X0 =  np.array([[1, 0],
    #              [0, 1]
    #              ], dtype='Float64')

    X0 = np.array([[1.000], [0.000],
                  [0.000], [1.000],
                  ], dtype='Float64')
    sys.X0 = np.reshape(X0, (2, 2))  # packed in matrix form
    #sys.X0 = np.reshape(X0, (2, 2))  # packed in matrix form

    M = linalg.sqrtm(sys.X0)

    #M = 0.5 * (M + M.T)
    #sys.M = np.reshape(0.5*(M+M.T),sys.n*sys.n,1)
    sys.M = 0.5 * (M + M.T)
    print "M: " + str(sys.M)
    # xl0 = M * l0

    #status check here, rank check


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
    center_trajectory = np.zeros((sys.n, len_prop), dtype=np.float)

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
        r.integrate(r.t + dt)

        try:
            center_trajectory[:,i] = r.y
        except Exception as e:
            if debug:
                print e
            print "trajectory_center[:,i] exited at " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1) + " dt: " + str(dt)
            return center_trajectory
        i += 1
    print "trajectory_center[:,i] exited at " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1) + " dt: " + str(dt)
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

def EA_reach_per_search(sys, evolve = False, y0 = [],debug = False):
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

    #dxdt = np.reshape(dxdt,n*n,1)
    dxdt = np.reshape(0.5*(dxdt+dxdt.T),n*n,1)


    return dxdt

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

    T = U1*V1*V2_T*U2_T


    #T = np.linalg.multi_dot([U1, V1, V2.T, U2.T])

    return T

def IA_reach_per_search(sys,evolve = False, y0 = [],debug = False):
    debug = True
    if evolve:
        print "evolving from latest tube result"
    else:
        #y0 = np.reshape(sys.X0, sys.n * sys.n)  # 3*3
        y0 = np.reshape(sys.M, (sys.n * sys.n), order='F')  # 3*3 #starting with half!

    t0 = sys.t0
    t1 = sys.t1
    dt = sys.dt

    len_prop = sys.len_prop
    if debug:
        print "len_prop: " + str(len_prop)

    #tube = np.zeros((sys.n * sys.n, len_prop), dtype=np.float)
    tube = np.zeros((sys.n * sys.n, len_prop), dtype='Float64')
    time_tube = np.zeros(len_prop, dtype=np.float)

    #
    # r = ode(deriv_ia_nodist).set_integrator('dop853',atol=1e-13,rtol=1e-13) #'dopri5'
    r = ode(deriv_ia_nodist).set_integrator('vode', atol=1e-13, rtol=1e-13)

    r.set_initial_value(y0, t0).set_f_params(sys.n)


    #tube[:, 0] = np.squeeze(y0)

    Q = np.reshape(y0, (sys.n, sys.n))
    tube[:, 0] = np.reshape(np.dot(Q, Q.T), (sys.n * sys.n), order='F')
    time_tube[0] = t0

    if debug:
        print "Y0, T0"
        print y0
        print t0
        print "END"
    i = 1
    while r.successful() and (r.t <= t1):
        print "starting loop with: " + str(r.t)
        # update_sys(sys, r.t, t0)  # update transition mat phi
        sys.F = linalg.expm((sys.A) * abs((r.t - t0)))
        print "sys.F: " + str(sys.F)
        # update sys
        r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)
        print "CHECK DXDT HERE"
        try:
            # append tube
            #ADDING FIX IESM
            #Q = fix_iesm(Q', d1);
            #Q' * Q

            #
            #Q*Q'
            ##np.squeeze(r.y)
            print "CHECK r.y: " + str(r.y)
            print "CHECK squeeeze(r.y): " + str(np.squeeze(r.y))

            #FIX IESM HERE
            #FIX_IESM - returns values for (Q' * Q).
            Q = np.reshape(r.y, (sys.n, sys.n))
            wrong_vec_q_alt = np.reshape(np.dot(Q.T, Q), sys.n * sys.n, order='F')

            vec_q = np.reshape(np.dot(Q,Q.T), sys.n * sys.n, order='F' ) #check the orientation
            print "vec_q: " +str(vec_q)
            #DSOHN fix
            #r.set_initial_value(vec_q)
            tube[:, i] = vec_q
            #tube[:, i] = np.squeeze(r.y)

            #validating L
            #sys.F = linalg.expm((sys.A) * abs((r.t - sys.t0)))
            #F = np.linalg.inv(sys.F)
            ##inv_F = np.linalg.inv(sys.F)  # verify THIS PROBABLY SINCE ADJOINT EQUATION IS -A
            ##F = np.linalg.inv(inv_F * sys.F) * inv_F
            #BPBsr = linalg.sqrtm(sys.BPB)
            #BPBsr = 0.5 * (BPBsr + BPBsr.T)
            ## Blocked Schur algorithms for computing the matrix square root
            #L0 = np.asmatrix(sys.L0).T
            #l = np.linalg.multi_dot([F.T, L0])
            #print l
            #print r.t
            time_tube[i] = r.t

        except Exception as e:
            if debug:
                print e
                print "tube[:,i] exited at " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1) + " dt: " + str(dt)
            return tube, time_tube
        if debug:
            print "tube_prop_number: " + str(i) + " r.t: " + str(r.t) + " t1: " + str(t1)
        i += 1
    if debug:
        print "finished at " + str(r.t)
    return tube, time_tube

def sys_set_for_debug(sys):
    xl0 = np.array([[1.00],
                       [0.00],
                       ], dtype='Float64')
    sys.xl0 =  np.reshape(xl0, (2, 1))  # packed in matrix form

    F = np.array([[0.9912], [0.3165],
                  [-0.0633], [1.2444],
                  ], dtype='Float64')
    sys.F = np.reshape(F, (2, 2)) #packed in matrix form

    L0 = np.array([[1.00],
                    [0.00],
                    ], dtype='Float64')
    sys.L0 = np.reshape(L0, (1, 2))  # packed in matrix form

    y = np.array([[1],
                   [0],
                   [0],
                   [1]]
                 , dtype='Float64')
    y = np.reshape(y, (4, 1))  # packed in matrix form
    return y

def deriv_ia_nodist(t,y,n,sys):
    print "calculate IA derive here!"
    #onetime
    #M = sqrtm(X0);
    #M = 0.5 * (M + M
    #');
    #xl0 = M * l0
    #dXdt = ell_iesm_ode(t, X, xl0, l0, mydata, n, back)
    #if hasdisturbance(lsys)
    #    [tt, Q] = ell_ode_solver( @ ell_iedist_ode, tvals, reshape(X0, d1 * d1, 1), l0, mydata, d1, back);
    #    Q = Q
    #    ';
    #elseif
    #~(isempty(mydata.BPB))
    #[tt, Q] = ell_ode_solver( @ ell_iesm_ode, tvals, reshape(M, d1 * d1, 1), M * l0, l0, mydata, d1, back);
    #Q = fix_iesm(Q
    #', d1);
    # "received" + str(y)

    #raw_xl0 = (sys.xl0)
#
    #raw_F = sys.F
#
    #raw_L0 = sys.L0
#
    #raw_y = y
#
    #y = sys_set_for_debug(sys)

    print "y0: " + str(y)
    X = np.reshape(y, (n, n), order='F')  # back to matrix
    A = sys.A

    #xl0 = sys.xl0
    #xl0 = sys.M*sys.L0

    #xl0 = np.asmatrix(np.linalg.multi_dot([sys.M,sys.L0])).T
    #print "before update: " +str(sys.F)
    sys.F = linalg.expm((sys.A) * abs((t - sys.t0)))
    #print "after update: " + str(sys.F)
    #F = np.linalg.inv(sys.F)
    F = np.linalg.inv(sys.F)  # verify THIS PROBABLY SINCE ADJOINT EQUATION IS -A
    #F = np.linalg.inv(inv_F*sys.F)*inv_F

    #from ell_inv
    #B = inv(A);
    #I = inv(B * A) * B;
    BPBsr =linalg.sqrtm(sys.BPB)
    BPBsr = 0.5*(BPBsr+BPBsr.T)
    #Blocked Schur algorithms for computing the matrix square root

    L0 = np.asmatrix(sys.L0).T

    L = np.linalg.multi_dot([F.T, L0])
    l = np.linalg.multi_dot([BPBsr, F.T, L0])
    xl0 = np.asmatrix(np.linalg.multi_dot([sys.M, L0]))
    # l = BPBsr * F.T * L0;

    if linalg.norm(l) < sys.abs_tol:
        S = np.eye(sys.n)
        print "linalg.norm(l) < sys.abs_tol"
    else:
        print "linalg.norm(l) > sys.abs_tol"
        l = np.asmatrix(l)#np.matrix('10; 0')
        #print "pre-SVD"
        #print xl0
        #print l.T
        #S = ell_valign(l, xl0)
        S = np.reshape(np.asmatrix(ell_valign(xl0, l)),(sys.n,sys.n), order='F')


    #dxdt_mat = np.asmatrix(np.dot(X, A.T))
    #dxdt_mat = np.asmatrix(np.dot(A,X))
    #dxdt_mat = np.matmul(A, X)
    #dxdt_mat = np.dot(X, A.T) + np.dot(S, BPBsr)
    dxdt_mat = np.asmatrix(np.linalg.multi_dot([X, A.T]))+np.asmatrix(np.linalg.multi_dot([S, BPBsr]))
    #dxdt_mat = np.asmatrix(np.linalg.multi_dot([X, A.T])) + BPBsr

    #dxdt_mat = np.asmatrix(np.linalg.multi_dot([X, A.T])) + BPBsr

    dxdt = np.reshape(dxdt_mat,(sys.n*sys.n), order='F')

    #dxdt = np.reshape(0.5*(dxdt_mat+dxdt_mat.T),sys.n*sys.n,1)

    print "t: " + str(t)
    print "y: " + str(y)
    print "X:" + str(X)
    #print "F: " + str(F)
    print "A: " + str(A)
    print "X*A.T: " + str(np.dot(X, A.T))

    print "l: " + str(l)
    print "BPBsr: " + str(BPBsr)
    print "S*BPBsr: " + str(np.linalg.multi_dot([S,BPBsr]))
    print "S: " + str(S)
    print "xl0: " + str(xl0)
    print "L: " + str(L)
    print dxdt

    return dxdt

def EA_evolve_nodist(prev_reach_set,time_tube,prev_center_trajectory,sys, extra_time,debug = False):
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
        tube, extra_time_tube = EA_reach_per_search(sys, evolve= True, y0 = prev_reach_set[:,-1,i]) #resume from last element
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
    sys = system_init(t_end = 0.5) #SETUP DYNAMIC SYSTEM DESCRIPTION HERE
    print "starting center traj calculation"
    center_trajectory = reach_center(sys)
    print "done center traj calculation"

    EA_reach_set =np.empty((sys.n*sys.n,sys.len_prop, sys.num_search))
    IA_reach_set = np.empty((sys.n * sys.n, sys.len_prop, sys.num_search))
    time_tube=[] #TIME_STAMP RECORD TO GUARANTEE CORRECT TIME SCALING OF THE GUI

    for i in range(sys.num_search):
        sys.L0 = sys.L[i].T

        #sys.L0 = sys.L[i].T
        #sys.L0 = np.reshape(sys.L0, (1, sys.n))  # packed in matrix form

        #EA_tube,time_tube = EA_reach_per_search(sys)
        print "sys.L0: " +str(sys.L0)
        sys.xl0 = np.asmatrix(np.linalg.multi_dot([sys.M,sys.L0])).T
        print "sys.xl0: " + str(sys.xl0)

        IA_tube,time_tube = IA_reach_per_search(sys)

        #EA_tube, time_tube = EA_reach_per_search(sys)

        #EA_reach_set[:,:,i] = EA_tube    #time_series x vectorized shape matrix x search direction
        IA_reach_set[:, :, i] = IA_tube
        #if debug:
            #print "tube shape: " + str(EA_tube.shape)
            #print EA_reach_set.shape
            #print "last of tube: "
            #print EA_tube[:, -1]
    #inspect_slice(reach_set,sys)
    ev.reach_gui(sys,center_trajectory,IA_reach_set,render_length=sys.len_prop,time_tube=time_tube)
    #ev.reach_gui(sys,center_trajectory,EA_reach_set,render_length=sys.len_prop,time_tube=time_tube)

    #print "EVOLVE 1"
    #evolved_reach_set, evolved_center_trajectory, evolved_time_tube = EA_evolve_nodist(reach_set,time_tube,center_trajectory,sys,extra_time=4)
#
    #print "EVOLVE 2"
    #evolved_reach_set, evolved_center_trajectory,evolved_time_tube = EA_evolve_nodist(evolved_reach_set,evolved_time_tube, evolved_center_trajectory, sys, extra_time=3)
##
    #ev.reach_gui(sys,evolved_center_trajectory,evolved_reach_set[:,:,:], render_length = sys.len_prop,time_tube = evolved_time_tube)

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




    #if back > 0
    #    t = -t;
    #    F = ell_value_extract(mydata.Phi, t, [n n]);
    #    s = -1;
    #else
    #    F = ell_value_extract(mydata.Phinv, t, [n n]);
    #    s = 1;
    #end
#
    #A = ell_value_extract(mydata.A, t, [n n]);
    #BPBsr = ell_value_extract(mydata.BPBsr, t, [n n]);
    #X = reshape(X, n, n);
#
    #l = BPBsr * F' * l0;
#
    #if norm(l) < ellOptions.abs_tol
    #    S = eye(n);
    #else
    #    S = ell_valign(xl0, l);
    #end
#
    #dXdt = reshape((s * X * A' + S*BPBsr), n*n, 1);
#
    #return;