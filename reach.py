import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy import linalg
from scipy import special, optimize
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matlab.engine

import sys


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
        self.xc =[]
        self.Bp = []

def triag_facets(epoints_num, points_num):
  td = np.arange(1,points_num+1)
  I =  np.arange(1,epoints_num )
  adtime     = td[0:len(td)-1].T
  ttime_data = (adtime * epoints_num)
  adtime = (adtime-1)*epoints_num
  adtime = np.tile(adtime, (epoints_num-1,1))
  Ie = np.tile(I.T,(points_num-1,1)).T
  Ie = Ie + adtime
  Ie = Ie.flatten('F').T
  #td         = 1:1:(points_num);
  #I          = transpose(1:1:(epoints_num-1));
  #adtime     = transpose(td(1:(end-1)));
  #ttime_data = adtime*epoints_num;
  #adtime     = (adtime.'-1)*epoints_num;
  #adtime     = adtime(ones(1, epoints_num-1), :);
  #Ie         = I(:, ones(1, points_num-1)) + adtime;
  #Ie         = Ie(:);
#
  a = np.vstack((Ie, Ie+1, Ie+1+epoints_num)).T
  b = np.vstack((Ie+1+epoints_num, Ie+epoints_num, Ie)).T
  c= np.vstack((ttime_data, ttime_data+1-epoints_num, ttime_data+1)).T
  d = np.vstack((ttime_data+1, ttime_data+epoints_num, ttime_data)).T


  facets = np.zeros(((points_num - 1) * epoints_num * 2, 3))

  for i in range(0,3):

    facets[:, i] = np.hstack((a[:,i],b[:,i],c[:,i],d[:,i]))

  return facets

def deriv_ea_nodist(t,y,n,sys):

    # "received" + str(y)
    X = np.reshape(y, (n, n)) #back to matrix
    A = sys.A
    #F = sys.F #gives complete bogus
    F = np.linalg.inv(sys.F) #verify THIS PROBABLY SINCE ADJOINT EQUATION IS -A
    BPB = sys.BPB

    L0 = sys.L0

    p1 = np.sqrt(np.linalg.multi_dot([L0.T, F, BPB, F.T, L0]))
    #print "TIME: " + str(t)
    #print "p1 " + str(p1)


    #print "F "+ str(F)
    #print "L0 " + str(L0)
    #print "X " + str(X)
    #print "y " + str(y)
    p2 = np.sqrt(np.linalg.multi_dot([L0.T, F, X, F.T, L0]))
    #print "p2 " + str(p2)

    if abs(p1) < sys.abs_tol:
        p1 = sys.abs_tol
    if abs(p2) < sys.abs_tol:
        p2 = sys.abs_tol


    pp1 = p1/p2

    #print "pp1 " + str(pp1)

    pp2 = p2/p1
    #print "pp2 " + str(pp2)
    dxdt = np.dot(A,X) + np.dot(X,A.T) + pp1*X + pp2*BPB

    #print "dxdt" + str(dxdt)


    dxdt = np.reshape(0.5*(dxdt+dxdt.T),n*n,1)

    #print "dxdt2" + str(dxdt)

    return dxdt

def update_sys(sys,t_now,t0):

    #print "t_now: " + str(t_now)
    #print "t0: " + str(t0)
    #print "sys.A"

    sys.F = linalg.expm((sys.A)*(t_now-t0))

    #print sys.A
    #print "sys.F"
    #print sys.F

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
    sys.t1 = 10
    sys.dt=0.01

    sys.len_prop = int(np.ceil(float((sys.t1 - sys.t0) / sys.dt))) + 2
    return sys

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
    #sys.xc = np.reshape(sys.xc, (2, 1))

    sys.Bp = np.array([[0],[0]])
    #sys.Bp = np.reshape(sys.Bp, (2, 1))

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

    #print "sys.BPB"

    #print sys.BPB

    sys.L = np.array([[1, 0],
                      [0, 1]
                      ])

    X0 =  np.array([[1, 0],
                  [0, 1]
                  ])

    #X0= np.array([[np.random.random_integers(1,5), 0, 0],
    #               [0, np.random.random_integers(1,5), 0],
    #               [0, 0, np.random.random_integers(1,5)]])

    sys.X0 = np.reshape(X0, (2, 2))  # packed in matrix form

    sys.num_search = 2
    sys.n = 2
    sys.abs_tol = 0.0001
    sys.t0 = 0.0
    sys.t1 = 10.0
    sys.dt=float(float(sys.t1-sys.t0))/200
    #sys.dt = 0.001
    print "sys dt " + str(sys.dt)
    sys.len_prop = int(np.ceil(float((sys.t1 - sys.t0) / sys.dt))) + 2
    return sys

def reach_vtx(sys,num_spokes_per_slice,unit_spokes,center_trajectory,reach_set):

    V = np.empty([3, 1])
    for t in range(sys.len_prop-1): #for each time prop
        #print t

        wheel_slice = np.zeros((2,num_spokes_per_slice), dtype=np.float)
        for spoke_index in range(num_spokes_per_slice):

            #print spoke_index
            spoke = unit_spokes[:,spoke_index]
            mval = sys.abs_tol #initializing spoke length max value search

            for ellipsoid_index in range(sys.num_search):
                #print ellipsoid_index

                #max spoke radius search (EA)
                #print np.reshape(reach_set[:,4, 1], (sys.n, sys.n))

                #INVERSION IS HAPPENING DUE TO DEFINITION OF ELLIPSOID USING SHAPE MATRIX
                Q = np.reshape(reach_set[:, t, ellipsoid_index], (sys.n, sys.n))
                Q = np.linalg.inv(Q)
                Q = Q.T
                #Q = np.reshape(reach_set[:,t, ellipsoid_index], (sys.n, sys.n))
                #Q = Q.T #as matlab does it...
                v = np.linalg.multi_dot([spoke.T, Q, spoke])
                if v > mval:
                    mval = v

            #finished getting max reach of this spoke
            adjusted_spoke = spoke/np.sqrt(mval)  #must make sure mval is set to abstol or else divide by 0
            #add center trajectory here
            wheel_slice[:,spoke_index] = adjusted_spoke
        #just finished wheel slice per time

        #create time stamp copies for slice points here [tt;X]
        current_time = t*sys.dt

        time_stamp = current_time*(np.ones(num_spokes_per_slice))
        #print "current time" + str(current_time)
        #print "wheel_slice"
        #print wheel_slice


        X = np.vstack((time_stamp, wheel_slice))

        #print X
        V = np.hstack((V,X))
        #print V

        #pdb.set_trace()
        #tt = rs.time_values(ii) * ones(1, num_spokes_per_slice);
        #    X = [tt; X];
        #    V = [V X];



    return V
def reach_gui(sys,center_trajectory, reach_set):
    print "generating vertices and facets"

    num_spokes_per_slice = 200
    phi = np.linspace(0,2*np.pi,num_spokes_per_slice)
    unit_spokes = np.array([np.cos(phi),np.sin(phi)])
    #print unit_spokes[0,:]
    #print unit_spokes[1, :]

    #print unit_spokes
    #print phi*180/np.pi

    vertices = reach_vtx(sys,num_spokes_per_slice,unit_spokes,center_trajectory,reach_set)
    #tube slice rendering vector numbers
    #sys.len_prop #time stamps
    F = triag_facets(num_spokes_per_slice, sys.len_prop)

    fig = plt.figure()
    ax = Axes3D(fig)
    #x = [0, 1, 1, 0]
    #y = [0, 0, 1, 1]
    #z = [0, 1, 0, 1]
    #verts = [zip(x, y, z)]
    verts = [zip(vertices[0,:],vertices[1,:],vertices[2,:])]
    #print vertices[0,:]

    #print verts
    #verts = [zip(x, y, z)]  # [(0,0,0), (1,0,1), (1,1,0), (0,1,1)]
    tri = Poly3DCollection(verts)  # Create polygons by connecting all of the vertices you have specified
    #tri.set_color(colors.rgb2hex(sp.rand(3)))  # Give the faces random colors
    tri.set_edgecolor('k')  # Color the edges of every polygon black
    ax.add_collection3d(tri)  # Connect polygon collection to the 3D axis
    plt.show()

    #ax.add_collection3d(Poly3DCollection(vertices))
    #plt.show()

    # eng = matlab.engine.start_matlab()

    # eng.triarea(nargout=0)

    #ret = eng.triarea(1.0, 5.0)

    #print vertices.shape
    #print F.shape

    #v_list = vertices.tolist()
    #print v_list
    #f_list = F.tolist()
    #print f_list
    #data_list = data.tolist()
    #a = [data_list{:}]; x = cell2mat(a);
    #y = double(reshape(x, 32, 32));
    #eng = matlab.engine.connect_matlab()
    #eng = matlab.engine.start_matlab()
    #ret = eng.plotforme(v_list,f_list)
    #print(ret)

    #render with vertices and facets.

    #print F

def reach_per_search(sys):
    #returns timeseries evolution of reachable set
    y0 = np.reshape(sys.X0, sys.n * sys.n) #3*3

    t0 = sys.t0
    t1 = sys.t1
    dt = sys.dt

    len_prop = sys.len_prop
    tube = np.zeros((sys.n*sys.n, len_prop), dtype=np.float)

    r = ode(deriv_ea_nodist).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(sys.n)

    tube[:,0] = y0

    i = 1
    while r.successful() and r.t <= t1:
        update_sys(sys, r.t, t0)  # update transition mat phi
        #print "sys.F should be changing"
        #print sys.F

        # update sys
        r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)
        tube[:,i] = r.y
        i += 1
        #print "iteration: " + str(i)

        # print("%g %g" % (r.t, r.y))
        # print r.t
        # print "result"
        # print np.reshape(r.y, (n, n))

    # print r.t
    #print "done"
    # print r.y
    #print np.reshape(r.y, (sys.n, sys.n))
    #print np.reshape(tube[:,-1], (sys.n, sys.n))
    print "finished at " + str(r.t)
    return tube
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
def reach_center(sys):
    #center calculation here

    #[tt, xx] = ell_ode_solver( @ ell_center_ode, tvals, x0, mydata, d1, back);

    #y0 = np.reshape(sys.xc, (sys.n,1))  #nx1
    y0 = np.reshape(sys.xc, (sys.n))
    t0 = sys.t0
    t1 = sys.t1
    dt = sys.dt

    len_prop = sys.len_prop
    center_trajectory = np.zeros((sys.n, len_prop), dtype=np.float)

    r = ode(deriv_reach_center).set_integrator('dopri5')
    r.set_initial_value(y0, t0).set_f_params(sys)

    center_trajectory[:, 0] = y0

    i = 1
    print "starting from" + str(r.y)
    while r.successful() and r.t <= t1:
        #update_sys(sys, r.t, t0)  # update transition mat phi
        # update sys
        #r.set_f_params(sys.n, sys)
        r.integrate(r.t + dt)
        center_trajectory[:,i] = r.y
        #print "r.t" + str(r.t)
        i += 1
        #print r.y

    return center_trajectory
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

def evolve_nodist(reach_set,sys, extra_time):
    switch_system(sys)

    #sys.num_search = 2
    #sys.n = 2
    #sys.abs_tol = 0.0001
    #sys.t0 = 0.0
    #sys.t1 = 10.0
    #sys.dt = float(float(sys.t1 - sys.t0)) / 200
    #sys.len_prop = int(np.ceil(float((sys.t1 - sys.t0) / sys.dt))) + 2

    #update system variables here

    #calculate extended portion of evolution per direction
    #concatenate

def reach():
    sys = system_init_LC()

    print "starting center traj calculation"
    center_trajectory = reach_center(sys)

    print center_trajectory.shape
    print center_trajectory[:,0]
    print center_trajectory[:,1]
    print "done center traj calculation"

    reach_set =np.empty((sys.n*sys.n, sys.len_prop, sys.num_search))
    #initialize the shape matrix tube

    for i in range(sys.num_search):
        sys.L0 = sys.L[i].T
        tube = reach_per_search(sys)
        reach_set[:,:,i] = tube    #time_series x vectorized shape matrix x search direction

    inspect_slice(reach_set,sys)

    #print "sys.BPB"
    #print sys.BPB
    #sample values here for debugging

    #graphing here after applying projection in orthogonal vector -> simply BB.T * shape matrix, where BB is normalized orthogonal basis

    print reach_set.shape
    #print np.reshape(reach_set[:,170,0], (sys.n, sys.n))

    #print np.reshape(reach_set[:,170, 1], (sys.n, sys.n))


    reach_gui(sys,center_trajectory,reach_set)
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

    try:
        print sys.argv[1]
    except:
        print " no sys arg "
    main()