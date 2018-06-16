import numpy as np
from scipy.integrate import ode
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



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
def reach_vtx(sys,num_spokes_per_slice,unit_spokes,center_trajectory,reach_set,render_length,time_tube,debug = False):

    V = np.empty([3, 1])
    for t in range(render_length-1): #for each time prop
        if debug:
            print "t: " + str(t)


        wheel_slice = np.zeros((2,num_spokes_per_slice), dtype=np.float)
        for spoke_index in range(num_spokes_per_slice):

            #print spoke_index
            spoke = unit_spokes[:,spoke_index]
            mval = sys.abs_tol #initializing spoke length max value search

            for ellipsoid_index in range(sys.num_search):
                #print "ellipsoidal_index: " + str(ellipsoid_index)

                #max spoke radius search (EA)
                #print np.reshape(reach_set[:,4, 1], (sys.n, sys.n))

                #INVERSION IS HAPPENING DUE TO DEFINITION OF ELLIPSOID USING SHAPE MATRIX

                Q = np.reshape(reach_set[:, t, ellipsoid_index], (sys.n, sys.n))
                Q = np.linalg.inv(Q)
                Q = Q.T

                #try:
                #    Q = np.reshape(reach_set[:, t, ellipsoid_index], (sys.n, sys.n))
                #    Q = np.linalg.inv(Q)
                #    Q = Q.T
                #except:
                #    print "SINGULAR Q"
                #    print "time: " + str(t)
                #    print Q
                #Q = np.reshape(reach_set[:,t, ellipsoid_index], (sys.n, sys.n))
                #Q = Q.T #as matlab does it...
                v = np.linalg.multi_dot([spoke.T, Q, spoke])
                if v > mval:
                    mval = v

            #finished getting max reach of this spoke
            adjusted_spoke = spoke/np.sqrt(mval)  #must make sure mval is set to abstol or else divide by 0
            #add center trajectory here
            wheel_slice[:,spoke_index] = adjusted_spoke + center_trajectory[:,t]
        #just finished wheel slice per time

        #create time stamp copies for slice points here [tt;X]
        #current_time = t*sys.dt #find correc dt here
        current_time = time_tube[t]
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
def reach_gui(sys,center_trajectory, reach_set,render_length,time_tube,debug = False):
    print "generating vertices and facets"

    num_spokes_per_slice = 200
    phi = np.linspace(0,2*np.pi,num_spokes_per_slice)
    unit_spokes = np.array([np.cos(phi),np.sin(phi)])
    vertices = reach_vtx(sys,num_spokes_per_slice,unit_spokes,center_trajectory,reach_set, render_length,time_tube)
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
